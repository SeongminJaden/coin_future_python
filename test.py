import time
import datetime
import csv
import os
from dotenv import load_dotenv
from gate_api import ApiClient, Configuration, FuturesApi
from gate_api.models import FuturesOrder
from decimal import Decimal as D, ROUND_DOWN
from gate_api.exceptions import GateApiException

# .env 파일에서 API 키 읽기
load_dotenv()

# API 키 설정 (환경변수에서 읽기 또는 직접 설정)
GATE_API_KEY = os.getenv("GATE_API_KEY")
GATE_API_SECRET = os.getenv("GATE_API_SECRET")

# API 키가 환경변수에 없으면 직접 설정 (실제 API 키로 교체하세요)
if not GATE_API_KEY or not GATE_API_SECRET:
    print("경고: .env 파일에서 API 키를 찾을 수 없습니다.")
    print("아래에 실제 API 키를 입력하거나 .env 파일을 생성하세요.")
    # GATE_API_KEY = "your_actual_api_key_here"
    # GATE_API_SECRET = "your_actual_api_secret_here"

# API 키가 없으면 테스트 모드로 실행
if not GATE_API_KEY or not GATE_API_SECRET:
    print("API 키가 설정되지 않았습니다. 공개 API만 사용합니다.")
    # 공개 API는 인증이 필요하지 않음
    configuration = Configuration()
else:
    configuration = Configuration(
        key=GATE_API_KEY,
        secret=GATE_API_SECRET,
    )

api_client = ApiClient(configuration)
futures_api = FuturesApi(api_client)

SYMBOL = "ETH_USDT"  # 거래 페어 (이더리움)
SETTLE = "usdt"  # 정산 통화
LEVERAGE = 20  # 레버리지 25배로 고정
EMA_PERIODS = [10, 20, 50]  # EMA 기간들
POSITION = None
LOG_FILE = None
LOG_FILE_START_HOUR = None
MIN_ORDER_SIZE = None  # 동적으로 설정될 예정

# 매매 설정
POSITION_SIZE_PERCENT = 0.9  # 포지션 규모 90% (지갑의 90%)
STOP_LOSS_PERCENT = 0.005  # 손절 0.5% (0.25-1% 범위)
TAKE_PROFIT_PERCENT = 0.01  # 익절 1%
BREAKOUT_THRESHOLD = 0.02  # 돌파 임계값 2%

# 수수료 설정
MAKER_FEE_RATE = 0.0001  # Maker 수수료 0.01%
TAKER_FEE_RATE = 0.00075  # Taker 수수료 0.075%
FUNDING_FEE_RATE = 0.0001  # 자금조달 수수료 (평균)

def calculate_ema(prices, period):
    """EMA 계산"""
    if len(prices) < period:
        return None
    
    multiplier = 2 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema

def get_current_emas():
    """EMA 10, 20, 50 계산"""
    try:
        candles = futures_api.list_futures_candlesticks(
            contract=SYMBOL,
            settle=SETTLE,
            interval="1m",
            limit=max(EMA_PERIODS) + 10,
        )
        
        # Gate.io API에서 캔들 데이터는 c 속성으로 접근
        closes = [float(c.c) for c in candles]
        
        emas = {}
        for period in EMA_PERIODS:
            emas[period] = calculate_ema(closes, period)
        
        return emas
    except Exception as e:
        log_message(f"EMA 계산 오류: {e}")
        return None

def get_latest_candles(limit=20):
    """최근 캔들 데이터 조회"""
    try:
        candles = futures_api.list_futures_candlesticks(
            contract=SYMBOL,
            settle=SETTLE,
            interval="1m",
            limit=limit,
        )
        return candles if candles else None
    except Exception as e:
        log_message(f"캔들 데이터 조회 오류: {e}")
        return None

def is_long_candle(candle, threshold=0.02):
    """긴 양봉인지 확인"""
    try:
        open_price = float(candle.o)
        close_price = float(candle.c)
        high_price = float(candle.h)
        low_price = float(candle.l)
        
        # 양봉인지 확인
        if close_price <= open_price:
            return False
        
        # 캔들 길이 계산 (고가-저가 대비 시가-종가 비율)
        candle_body = abs(close_price - open_price)
        candle_range = high_price - low_price
        
        if candle_range == 0:
            return False
        
        body_ratio = candle_body / candle_range
        return body_ratio > threshold
        
    except Exception as e:
        log_message(f"캔들 분석 오류: {e}")
        return False

def is_short_candle(candle, threshold=0.02):
    """긴 음봉인지 확인"""
    try:
        open_price = float(candle.o)
        close_price = float(candle.c)
        high_price = float(candle.h)
        low_price = float(candle.l)
        
        # 음봉인지 확인
        if close_price >= open_price:
            return False
        
        # 캔들 길이 계산 (고가-저가 대비 시가-종가 비율)
        candle_body = abs(close_price - open_price)
        candle_range = high_price - low_price
        
        if candle_range == 0:
            return False
        
        body_ratio = candle_body / candle_range
        return body_ratio > threshold
        
    except Exception as e:
        log_message(f"캔들 분석 오류: {e}")
        return False

def is_sideways_market(candles, emas):
    """횡보장인지 확인 (EMA들이 서로 가까이 있을 때)"""
    if not emas or len(candles) < 10:
        return False
    
    ema_10 = emas.get(10)
    ema_20 = emas.get(20)
    ema_50 = emas.get(50)
    
    if not all([ema_10, ema_20, ema_50]):
        return False
    
    # EMA들이 서로 1% 이내에 있는지 확인
    avg_ema = (ema_10 + ema_20 + ema_50) / 3
    tolerance = avg_ema * 0.01  # 1% 허용 오차
    
    return (abs(ema_10 - ema_20) < tolerance and 
            abs(ema_20 - ema_50) < tolerance and 
            abs(ema_10 - ema_50) < tolerance)

def is_breakout_condition(candles, emas):
    """돌파 조건 확인: 횡보 후 EMA 위에서 긴 양봉"""
    if not candles or len(candles) < 20:
        return False
    
    # 최근 10개 캔들로 횡보장 확인
    recent_candles = candles[:10]
    if not is_sideways_market(recent_candles, emas):
        return False
    
    # 최신 캔들이 긴 양봉인지 확인
    latest_candle = candles[0]
    if not is_long_candle(latest_candle):
        return False
    
    # 최신 캔들이 EMA 위에 있는지 확인
    close_price = float(latest_candle.c)
    ema_10 = emas.get(10)
    ema_20 = emas.get(20)
    
    if not ema_10 or not ema_20:
        return False
    
    # EMA 10과 20 위에서 돌파
    return close_price > ema_10 and close_price > ema_20

def is_breakdown_condition(candles, emas):
    """하락 돌파 조건 확인: 횡보 후 EMA 아래에서 긴 음봉"""
    if not candles or len(candles) < 20:
        return False
    
    # 최근 10개 캔들로 횡보장 확인
    recent_candles = candles[:10]
    if not is_sideways_market(recent_candles, emas):
        return False
    
    # 최신 캔들이 긴 음봉인지 확인
    latest_candle = candles[0]
    if not is_short_candle(latest_candle):
        return False
    
    # 최신 캔들이 EMA 아래에 있는지 확인
    close_price = float(latest_candle.c)
    ema_10 = emas.get(10)
    ema_20 = emas.get(20)
    
    if not ema_10 or not ema_20:
        return False
    
    # EMA 10과 20 아래에서 하락 돌파
    return close_price < ema_10 and close_price < ema_20

def list_futures_positions():
    # 인증이 필요한 함수이므로 테스트 모드에서는 None 반환
    if not GATE_API_KEY or not GATE_API_SECRET:
        return None
    
    try:
        # 방법 1: list_positions 사용
        positions = futures_api.list_positions(settle=SETTLE)
        
        if positions:
            for p in positions:
                if p.contract == SYMBOL and abs(p.size) > 0.001:
                    return p
        
        # 방법 2: get_position 사용 (특정 계약의 포지션 조회)
        try:
            position = futures_api.get_position(settle=SETTLE, contract=SYMBOL)
            if position and abs(position.size) > 0.001:
                return position
        except Exception as e2:
            pass
        
        return None
    except Exception as e:
        return None

def log_message(msg, is_trade=False, is_status=False):
    global LOG_FILE, LOG_FILE_START_HOUR
    now = datetime.datetime.now()
    
    # 거래 관련 로그만 CSV에 저장
    if is_trade:
        if LOG_FILE is None or now.hour != LOG_FILE_START_HOUR:
            LOG_FILE_START_HOUR = now.hour
            log_filename = f"거래로그_{now.strftime('%Y%m%d_%H')}.csv"
            if LOG_FILE:
                LOG_FILE.close()
            LOG_FILE = open(log_filename, mode='a', newline='', encoding='utf-8-sig')
            writer = csv.writer(LOG_FILE)
            if os.stat(log_filename).st_size == 0:
                writer.writerow(["시간", "메시지"])
            print(f"[로그] 새 로그파일 생성: {log_filename}")
        writer = csv.writer(LOG_FILE)
        writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), msg])
        LOG_FILE.flush()
    
    # 거래 관련 로그와 상태 로그만 콘솔에 출력
    if is_trade or is_status:
        print(f"[로그] {msg}")

def set_leverage():
    """레버리지를 설정합니다."""
    if not GATE_API_KEY or not GATE_API_SECRET:
        return None
    
    try:
        # 먼저 현재 포지션 확인
        position = list_futures_positions()
        
        # 포지션이 있고 크기가 0이 아닐 때만 건너뛰기
        if position is not None and hasattr(position, 'size') and abs(position.size) > 0.001:
            return None
        
        # 포지션이 없거나 매우 작을 때만 레버리지 설정
        res = futures_api.update_position_leverage(
            settle=SETTLE,
            contract=SYMBOL,
            leverage=LEVERAGE
        )
        return res
    except Exception as e:
        return None

def get_contract_info():
    """계약 정보를 조회하여 최소 주문 단위 등을 확인"""
    try:
        contract_info = futures_api.get_futures_contract(SETTLE, SYMBOL)
        log_message(f"계약 정보: 최소주문={contract_info.order_size_min}, 최대주문={contract_info.order_size_max}, 승수={contract_info.quanto_multiplier}", is_status=True)
        return contract_info
    except Exception as e:
        log_message(f"계약 정보 조회 실패: {e}", is_status=True)
        return None

def create_order(margin_usdt, trade_side):
    # 인증이 필요한 함수이므로 테스트 모드에서는 로그만 출력
    if not GATE_API_KEY or not GATE_API_SECRET:
        return None
    
    try:
        # 1. 현재 레버리지 설정
        futures_api.update_position_leverage(SETTLE, SYMBOL, str(LEVERAGE))
        
        # 2. ETH 계약 정보, 현재 가격 불러오기
        contract_info = futures_api.get_futures_contract(SETTLE, SYMBOL)
        multiplier = D(contract_info.quanto_multiplier)
        
        tickers = futures_api.list_futures_tickers(SETTLE, contract=SYMBOL)
        if not tickers:
            log_message("ETH_USDT 가격 정보를 불러오지 못했습니다.", is_trade=True)
            return None
        price = D(tickers[0].last)
        
        # 3. 주문 사이즈 계산 (마진 기반)
        size = (D(margin_usdt) * D(LEVERAGE)) / (price * multiplier)
        size = size.to_integral_value(rounding=ROUND_DOWN)
        
        # 숏 포지션이면 size 음수로
        if trade_side == "sell":
            size = -size
        
        # 4. 증거금보다 잔고 부족하면 spot → futures로 전송
        try:
            account = futures_api.list_futures_accounts(SETTLE)
            available = D(account.available)
        except GateApiException as ex:
            if ex.label == "USER_NOT_FOUND":
                available = D(0)
            else:
                raise ex
        
        required_margin = size.copy_abs() * price * multiplier / D(LEVERAGE) * D("1.1")
        if available < required_margin:
            log_message(f"잔고 부족. 필요: {required_margin}, 가능: {available}", is_trade=True)
            return None
        
        # 5. 기존 주문 취소 후 주문 실행
        futures_api.cancel_futures_orders(SETTLE, SYMBOL)
        
        order = FuturesOrder(
            contract=SYMBOL,
            size=int(size),
            price="0",  # 시장가 주문
            tif="ioc"
        )
        
        res = futures_api.create_futures_order(SETTLE, order)
        log_message(f"{trade_side} 주문 체결: {res}", is_trade=True)
        return res
        
    except Exception as e:
        log_message(f"주문 실패: {e}", is_trade=True)
        return None

def close_position(position, partial_close=False):
    if position is None or position.size == 0:
        return
    
    # 현재 가격 가져오기
    try:
        candles = futures_api.list_futures_candlesticks(
            contract=SYMBOL,
            settle=SETTLE,
            interval="1m",
            limit=1,
        )
        current_price = float(candles[0].c) if candles else 2400
    except:
        current_price = 2400  # 기본값
    
    # Gate.io 공식 예제와 동일하게 size의 양수/음수로 방향 구분
    if position.size > 0:
        # 롱 포지션 청산: 음수 size로 주문
        order_size = -abs(position.size)
        trade_side = "sell"
    else:
        # 숏 포지션 청산: 양수 size로 주문
        order_size = abs(position.size)
        trade_side = "buy"
    
    # 부분 청산이 필요한 경우
    if partial_close:
        available_balance = get_available_balance()
        
        # 레버리지 5배 기준으로 실제 청산 가능한 수량 계산
        # 마진 = 주문가격 * 수량 / 레버리지
        # 수량 = 마진 * 레버리지 / 주문가격
        max_margin = available_balance * 0.95  # 95% 사용 (여유분 확보)
        max_order_size = (max_margin * LEVERAGE) / current_price
        
        # 최소 주문 단위 (0.01 ETH) 체크
        if max_order_size < 0.01:
            log_message(f"잔고가 부족하여 부분 청산도 불가능합니다. (필요: 0.01 ETH, 가능: {max_order_size:.2f} ETH)", is_trade=True)
            return None
        
        # 실제 청산 수량 결정
        actual_order_size = min(abs(position.size), max_order_size)
        if position.size > 0:
            order_size = -actual_order_size
        else:
            order_size = actual_order_size
            
        log_message(f"부분 청산 시도: {trade_side} {actual_order_size} ETH (전체: {abs(position.size)} ETH, 잔고: {available_balance:.2f} USDT)", is_trade=True)
    else:
        log_message(f"포지션 청산 시도: {trade_side} {abs(position.size)} ETH", is_trade=True)
    
    # Gate.io 공식 예제와 동일한 파라미터 사용 (reduce_only 추가)
    order = FuturesOrder(
        contract=SYMBOL,
        size=order_size,
        price="0",  # 시장가 주문
        tif='ioc',   # 즉시 체결 또는 취소
        reduce_only=True  # 기존 포지션만 줄이기
    )
    
    try:
        res = futures_api.create_futures_order(
            settle=SETTLE,
            futures_order=order
        )
        if partial_close:
            log_message(f"부분 청산 완료: {res}", is_trade=True)
        else:
            log_message(f"포지션 청산 완료: {res}", is_trade=True)
        return res
    except Exception as e:
        if partial_close:
            log_message(f"부분 청산 실패: {e}", is_trade=True)
        else:
            log_message(f"포지션 청산 실패: {e}", is_trade=True)
        return None

def get_available_balance():
    if not GATE_API_KEY or not GATE_API_SECRET:
        return 0
    try:
        accounts = futures_api.list_futures_accounts(settle=SETTLE)
        # USDT 선물 계좌의 사용 가능 잔고(available) 추출
        if isinstance(accounts, list):
            for acc in accounts:
                if hasattr(acc, 'available'):
                    return float(acc.available)
        elif hasattr(accounts, 'available'):
            return float(accounts.available)
        return 0
    except Exception as e:
        log_message(f"잔고 조회 실패: {e}")
        return 0

def log_position_info():
    """현재 포지션 정보를 로그로 출력"""
    try:
        position = list_futures_positions()
        if not position or not hasattr(position, 'size'):
            return
        
        size = float(position.size)
        
        if size == 0:
            return
        
        entry_price = float(position.entry_price if hasattr(position, 'entry_price') else 0)
        
        # 현재가 직접 조회
        candles = get_latest_candles(limit=1)
        current_price = float(candles[0].c) if candles else 0
        
        if current_price and entry_price > 0:
            if size > 0:  # 롱 포지션
                profit_pct = (current_price - entry_price) / entry_price * 100
                position_type = "롱"
            else:  # 숏 포지션
                profit_pct = (entry_price - current_price) / entry_price * 100
                position_type = "숏"
            
            log_message(f"현재 {position_type} 포지션: {abs(size):.2f} ETH, 진입가: {entry_price:.2f}, 현재가: {current_price:.2f}, 수익률: {profit_pct:.2f}%")
        
    except Exception as e:
        log_message(f"포지션 정보 로그 오류: {e}")

def calculate_total_fees(position_size, current_price):
    """포지션의 총 수수료 계산 (진입 + 청산 + 자금조달)"""
    position_value = abs(position_size) * current_price
    
    # 진입 수수료 (Taker)
    entry_fee = position_value * TAKER_FEE_RATE
    
    # 청산 수수료 (Taker)
    exit_fee = position_value * TAKER_FEE_RATE
    
    # 자금조달 수수료 (8시간마다, 하루 3회 가정)
    funding_fee = position_value * FUNDING_FEE_RATE * 3
    
    total_fees = entry_fee + exit_fee + funding_fee
    return total_fees

def is_profitable_after_fees(position, current_price, ema_20):
    """수수료를 고려한 실제 수익 여부 확인"""
    if position is None or abs(position.size) < 0.001:
        return False
    
    # 진입가 계산
    entry_price = float(position.entry_price) if hasattr(position, 'entry_price') else current_price
    
    # 총 수수료 계산
    total_fees = calculate_total_fees(position.size, current_price)
    
    # 수익 계산
    if position.size > 0:  # 롱 포지션
        # EMA 20 이탈 조건 확인
        if current_price < ema_20:
            gross_profit = abs(position.size) * (current_price - entry_price)
            net_profit = gross_profit - total_fees
            return net_profit > 0
    else:  # 숏 포지션
        # EMA 20 이탈 조건 확인
        if current_price > ema_20:
            gross_profit = abs(position.size) * (entry_price - current_price)
            net_profit = gross_profit - total_fees
            return net_profit > 0
    
    return False

def trading_logic():
    global POSITION
    emas = get_current_emas()
    candles = get_latest_candles()
    position = list_futures_positions()
    POSITION = position

    if candles is None or emas is None:
        return

    # 포지션 상태 확인 (더 엄격한 체크)
    has_position = False
    if position is not None and hasattr(position, 'size'):
        if abs(position.size) > 0.001:  # 포지션이 실제로 있는 경우
            has_position = True

    # 포지션이 있을 때는 익절/손절 조건만 확인하고 즉시 리턴
    if has_position:
        current_price = float(candles[0].c)
        ema_10 = emas.get(10)
        ema_20 = emas.get(20)
        ema_50 = emas.get(50)
        
        if not ema_10 or not ema_20 or not ema_50:
            return
        
        # 롱 포지션 관리
        if position.size > 0:
            # 손절: EMA 10 아래 0.5% 지점
            stop_loss_price = ema_10 * 0.995
            if current_price <= stop_loss_price:
                log_message(f"롱 포지션 손절 실행 (가격: {current_price}, 손절가: {stop_loss_price})", is_trade=True)
                close_position()
                return
            
            # 익절: EMA 20 또는 EMA 50 도달 시 (수수료 고려)
            take_profit_price = min(ema_20, ema_50)
            if current_price >= take_profit_price:
                # 수수료를 고려한 실제 수익 계산
                entry_price = float(position.entry_price if hasattr(position, 'entry_price') else 0)
                if entry_price > 0:
                    gross_profit_pct = (current_price - entry_price) / entry_price
                    net_profit_pct = gross_profit_pct - calculate_total_fees(position.size, current_price)
                    
                    if net_profit_pct > 0:
                        log_message(f"롱 포지션 익절 실행 (가격: {current_price}, 익절가: {take_profit_price}, 수익률: {net_profit_pct:.2%})", is_trade=True)
                        close_position()
                        return
                    else:
                        log_message(f"롱 포지션 익절 조건이지만 수수료로 인해 손실 - 유지 (수익률: {net_profit_pct:.2%})")
        
        # 숏 포지션 관리
        elif position.size < 0:
            # 손절: EMA 10 위 0.5% 지점
            stop_loss_price = ema_10 * 1.005
            if current_price >= stop_loss_price:
                log_message(f"숏 포지션 손절 실행 (가격: {current_price}, 손절가: {stop_loss_price})", is_trade=True)
                close_position()
                return
            
            # 익절: EMA 20 또는 EMA 50 도달 시 (수수료 고려)
            take_profit_price = max(ema_20, ema_50)
            if current_price <= take_profit_price:
                # 수수료를 고려한 실제 수익 계산
                entry_price = float(position.entry_price if hasattr(position, 'entry_price') else 0)
                if entry_price > 0:
                    gross_profit_pct = (entry_price - current_price) / entry_price
                    net_profit_pct = gross_profit_pct - calculate_total_fees(position.size, current_price)
                    
                    if net_profit_pct > 0:
                        log_message(f"숏 포지션 익절 실행 (가격: {current_price}, 익절가: {take_profit_price}, 수익률: {net_profit_pct:.2%})", is_trade=True)
                        close_position()
                        return
                    else:
                        log_message(f"숏 포지션 익절 조건이지만 수수료로 인해 손실 - 유지 (수익률: {net_profit_pct:.2%})")
        
        return  # 포지션이 있지만 조건 미충족 시 조용히 리턴
    
    # 포지션이 없을 때만 새로운 진입 시도
    # 돌파 조건 확인
    if not is_breakout_condition(candles, emas) and not is_breakdown_condition(candles, emas):
        return
    
    available_balance = get_available_balance()
    if available_balance <= 0:
        return
        
    # 새로운 포지션을 열기 전에 레버리지 설정
    set_leverage()
    
    # 포지션 규모 계산 (지갑의 100%)
    order_value = available_balance * POSITION_SIZE_PERCENT
    
    # 롱 진입 (횡보 후 EMA 위에서 긴 양봉 돌파)
    if is_breakout_condition(candles, emas):
        log_message(f"상승 돌파 조건 충족! 롱 진입 (주문금액: {order_value:.2f} USDT)", is_trade=True)
        create_order(margin_usdt=order_value, trade_side="buy")
        log_position_info()  # 진입 후 포지션 정보 로그
    
    # 숏 진입 (횡보 후 EMA 아래에서 긴 음봉 하락 돌파)
    elif is_breakdown_condition(candles, emas):
        log_message(f"하락 돌파 조건 충족! 숏 진입 (주문금액: {order_value:.2f} USDT)", is_trade=True)
        create_order(margin_usdt=order_value, trade_side="sell")
        log_position_info()  # 진입 후 포지션 정보 로그

def test_api_connection():
    """API 연결 테스트"""
    try:
        # 간단한 API 호출 테스트
        log_message("자동매매 시작", is_status=True)
        
        # 캔들 데이터 가져오기 테스트
        candles = futures_api.list_futures_candlesticks(
            contract=SYMBOL,
            settle=SETTLE,
            interval="1m",
            limit=5,
        )
        
        if candles:
            log_message("API 연결 성공! 자동매매 준비 완료", is_status=True)
            
            # EMA 계산 테스트
            emas = get_current_emas()
            if emas:
                log_message(f"EMA 계산 성공: EMA10={emas.get(10):.2f}, EMA20={emas.get(20):.2f}, EMA50={emas.get(50):.2f}", is_status=True)
            else:
                log_message("EMA 계산 실패", is_status=True)
                
            log_position_info()  # 시작 시 현재 포지션 정보 출력
        else:
            log_message("캔들 데이터가 없습니다.", is_status=True)
            
    except Exception as e:
        log_message(f"API 연결 테스트 실패: {e}", is_status=True)

def get_position_pnl():
    """현재 포지션의 수익률 계산"""
    if not GATE_API_KEY or not GATE_API_SECRET:
        return 0, 0
    
    try:
        position = list_futures_positions()
        if position is None or abs(position.size) < 0.001:
            return 0, 0
        
        # 현재 가격 가져오기
        candles = futures_api.list_futures_candlesticks(
            contract=SYMBOL,
            settle=SETTLE,
            interval="1m",
            limit=1,
        )
        current_price = float(candles[0].c) if candles else 0
        
        if current_price == 0:
            return 0, 0
        
        # 진입가 계산 (position 객체에서 가져오거나 현재가 사용)
        entry_price = float(position.entry_price) if hasattr(position, 'entry_price') else current_price
        
        # 수익률 계산
        if position.size > 0:  # 롱 포지션
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
        else:  # 숏 포지션
            pnl_percent = ((entry_price - current_price) / entry_price) * 100
        
        # 절대 수익/손실 (USDT)
        pnl_usdt = abs(position.size) * (current_price - entry_price) if position.size > 0 else abs(position.size) * (entry_price - current_price)
        
        return pnl_percent, pnl_usdt
        
    except Exception as e:
        return 0, 0

def log_status():
    """현재 상태를 로그로 출력 (3초마다 호출)"""
    try:
        # USDT 잔고 조회
        available_balance = get_available_balance()
        
        # 포지션 정보 조회
        position = list_futures_positions()
        position_info = "없음"
        profit_pct = 0
        
        if position and hasattr(position, 'size'):
            size = float(position.size)
            
            if size != 0:
                entry_price = float(position.entry_price if hasattr(position, 'entry_price') else 0)
                
                # 현재가 직접 조회
                candles = get_latest_candles(limit=1)
                current_price = float(candles[0].c) if candles else 0
                
                if current_price and entry_price > 0:
                    if size > 0:  # 롱 포지션
                        profit_pct = (current_price - entry_price) / entry_price * 100
                        position_info = f"롱 {abs(size):.2f} ETH"
                    else:  # 숏 포지션
                        profit_pct = (entry_price - current_price) / entry_price * 100
                        position_info = f"숏 {abs(size):.2f} ETH"
        
        log_message(f"상태: USDT {available_balance:.2f} | 포지션: {position_info} | 수익률: {profit_pct:.2f}%", is_status=True)
        
    except Exception as e:
        log_message(f"상태 로그 오류: {e}", is_status=True)

def main():
    log_message("봇 시작", is_status=True)
    
    # 계약 정보 조회
    contract_info = get_contract_info()
    
    # 레버리지 설정
    set_leverage()
    
    # 시작 시 현재 잔고 출력
    available_balance = get_available_balance()
    log_message(f"현재 USDT 잔고: {available_balance:.2f} USDT", is_status=True)
    
    while True:
        try:
            trading_logic()
            # 봇 상태 메시지 (3초마다)
            if int(time.time()) % 3 == 0:
                log_status()
        except Exception as e:
            log_message(f"오류 발생: {e}", is_status=True)
        time.sleep(1)

if __name__ == "__main__":
    main()
