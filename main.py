# Project "Trading with Tinkoff API based on orderbook data"

'''
This code uses asyncio module to run price streaming, predicting price movement and trading simultaneously.
'''
import asyncio
import numpy as np
import time
import datetime as dt
import tinvest as ti
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import sklearn
import joblib as jl

# import warnings
# from sklearn.exceptions import UserWarning
# warnings.filterwarnings(action='ignore', category=UserWarning)

# Creating class that only has property "status". Will be helpful later. Definitely not the best approach but it works.
class SmthWithStatus():
    def __init__(self):
        self.status = None

# Specifying parameters for Tinkoff API and other initial parameters
TOKEN_SANDBOX =  # Your sandbox token
# 'BBG001Y2XS07' # ABNB # 'BBG003PHHZT1' # MRNA
FIGI = 'BBG00HTN2CQ3'  # SPCE # Identifier of instrument (stock) to trade
broker_iis_account_id_sandbox =  # Your broker iis sandbox account id
broker_account_id_sandbox =  # Your broker sandbox account id
broker_account_id_real =  # Your broker account id
candle_resolution = ti.CandleResolution.min1 # Timeframe of candle
duration = 3600 * 0.25  # How long to run the code (event loop) in seconds
sample_time = 60  # Time during which orderbooks are collected into sample
period = 3  # Period with which orderbooks are collected
depth = 20  # Depth of orderbook
num_to_group = 3  # Number of orderbook lines to group
lag = 0.2  # Time between new sample creation and start of prediction
bank_size = 200  # Amount of money to operate with
bound, tp_bound, sl_bound, bound_min = 0.02, 0.004, 0.005, -0.004 # Portions of price at the start of trade.
# If we multiply (1 + tp_bound) with start price we get preferred sell price (take-profit price).
# If we multiply (1 - sl_bound) with start price we get sell price to stop loss.

# Check if there is a whole number of periods in sample_time
assert sample_time % period == 0, "Number of periods in sample_time must be whole"

bad_minutes = [240] + list(range(417, 422)) + list(range(807, 812))  # Indices of minutes not to trade in

# Loading XGBoostClassifier model
xgbc = jl.load("Models/XGBC/xgbc_625.jl")

# Making lists with sample column indices
#price_cols = [41*k for k in range(20)]
#orderbook_cols = [x for x in list(range(820)) if x not in price_cols]
price_cols = [15*k for k in range(20)]
orderbook_cols = [x for x in list(range(300)) if x not in price_cols]


# Streaming coroutine. Creates live changing pool of the last values of candles and orderbooks
async def stream(token, figi, collect_duration, candle_resolution, depth, countdown=60-time.gmtime().tm_sec-time.time()%1):
    # Initializing global variable that contains current candle, orderbook and instrument_info
    global cur_events
    cur_events = np.zeros((3), dtype=object)
    start_time = time.time()

    # Starting stream how it's recommended in Tinkoff API SDK https://github.com/daxartio/tinvest
    async with ti.Streaming(token) as streaming:
        try:
            await streaming.candle.subscribe(figi, candle_resolution)
            await streaming.orderbook.subscribe(figi, depth)
            await streaming.instrument_info.subscribe(figi)
            async for event in streaming:
                # Updating cur_events
                if str(event.event) == 'Event.candle':
                    cur_events[0] = event
                elif str(event.event) == 'Event.orderbook':
                    cur_events[1] = event
                elif str(event.event) == 'Event.instrument_info':
                    cur_events[2] = event
                if (time.time() - start_time) >= (collect_duration + countdown + 5):
                    await streaming.stop()
        except asyncio.TimeoutError:
            print('Stream stopped!')


# Collecting coroutine. Collects data from current events provided by concurrent streaming
async def collect(duration, depth, sample_time, period, countdown=60 - time.gmtime().tm_sec - time.time() % 1):
    global X, X_part, gmtimes, y_prediction, y_prediction_min, y_pred, y_pred_proba
    X, X_parts, gmtimes = [], [], []
    y_pred = 0.0
    await asyncio.sleep(countdown)  # Waiting for start
    current = 0  # Initializing current time variable. When initialized equals 0, later will be changed
    sample_num = 0
    parts_number = int(round(sample_time / period))  # Specifying the number of parts sample consists of
    sample = np.zeros(((2 * depth + 1) * parts_number))  # Initializing sample ndarray with zeros
    start_time = time.time()
    pivot_time = round(time.time())
    while time.time() <= start_time + duration:
        if cur_events[2].payload.trade_status == 'normal_trading':
            if current != 0:
                last = round(current)
            else:
                last = time.time() - period
            gmtime = dt.datetime.utcnow()
            current = time.time()
            # Collecting price, bids, asks
            cur_price = np.array(cur_events[0].payload.c)
            cur_bids = np.array(cur_events[1].payload.bids)[:, 1]
            cur_asks = np.array(cur_events[1].payload.asks)[:, 1]
            # Creating a part of sample by stacking price, bids and asks together
            X_part = np.hstack([cur_price,
                                np.array([np.sum(cur_bids[i * num_to_group:np.min([i * num_to_group + num_to_group, depth])])\
                                for i in range(np.int32(np.ceil(depth / num_to_group)))], dtype=np.float64),
                                np.array([np.sum(cur_asks[i * num_to_group:np.min([i * num_to_group + num_to_group, depth])])\
                                for i in range(np.int32(np.ceil(depth / num_to_group)))], dtype=np.float64)
                                ])
            # Adding the part to parts list if has appropriate length
            if len(X_part) == (2 * np.int32(np.ceil(depth / num_to_group))) + 1:
                X_parts.append(X_part)
                gmtimes.append(gmtime)
            # If number of collected parts is enough for full sample then predict
            if len(X_parts) >= parts_number:
                # Checking if time is correct
                if (((gmtimes[-1] - gmtimes[(-1) * parts_number]) < dt.timedelta(seconds=(sample_time - period + 1))) &
                    ((gmtimes[-1] - gmtimes[(-1) * parts_number]) > dt.timedelta(seconds=(sample_time - period - 1)))):
                    # Creating sample, normalizing it and predicting price move
                    sample = np.hstack(X_parts[(-1) * parts_number:])
                    sample_normalized = np.zeros_like(sample)
                    sample_normalized[price_cols] = np.array(sample[price_cols], dtype=np.float64) / np.float64(sample[price_cols[-1]])
                    sample_normalized[orderbook_cols] = np.array(sample[orderbook_cols], dtype=np.float64)\
                                                        / np.average(np.array(sample[orderbook_cols], dtype=np.float64))
                    y_pred = xgbc.predict(np.array([sample_normalized]))[0]
                    y_pred_proba = xgbc.predict_proba(np.array([sample_normalized]))[0, 1]
                    print('Time: {}. Sample #{}. Prediction: {:.4f}. Worktime: {:.4f}.'\
                          .format(dt.datetime.utcnow(), sample_num, y_pred_proba, time.time() - current))
                    sample_num += 1
            if sample_num > 3:
                X_parts = X_parts[(-2) * parts_number:]
                gmtimes = gmtimes[(-2) * parts_number:]
        else:
            print('Not a good time for trading')
        pivot_time = pivot_time + period
        if pivot_time - time.time() < 0:
            print('Time Error. Predict mode')
        # Waitng a period of time for next iteration
        await asyncio.sleep(pivot_time - time.time())
    print('Collection complete!')


# Coroutine that buys when it has a signal and then sells when it reaches "take-profit" or "stop-loss"
async def trade(token, broker_account_id, figi, bank_size, duration, sample_time, period, lag, bound, tp_bound,
                sl_bound, countdown=60-time.gmtime().tm_sec-time.time()%1):
    # Waiting for start
    await asyncio.sleep(countdown + sample_time + lag)
    start_time = time.time()
    pivot_time = time.time()
    # Creating list of pivot times
    pivot_times = [start_time]
    t = start_time
    while t <= (start_time + duration - sample_time):
        t += period
        pivot_times.append(t)
    pivot_times = np.array(pivot_times)
    # Creating variables that will contain info about current trade state
    trade_status, order_placed, tp_placed = False, False, False
    buy_period = 1
    sell_period = 1
    execute_time = 20
    async with ti.AsyncClient(token, use_sandbox=True) as aclient:
        while (time.time() <= (start_time + duration - sample_time)):
            # Try to buy if prediction is good, there is no already placed order and current time is good
            if ((y_pred_proba >= bound) & (order_placed == False)) & (dt.datetime.utcnow().hour * 60 + dt.datetime.utcnow().minute not in bad_minutes):
                print('Time: %s. Buy Mode. Current price: %f' % (dt.datetime.utcnow(), np.round(np.float64(cur_events[0].payload.c), 2)))
                # Try to get portfolio
                try:
                    get_portfolio = await aclient.get_portfolio(broker_account_id)
                    portfolio = np.array([[position.instrument_type, position.figi, position.ticker, position.balance, position.lots]
                                      for position in get_portfolio.payload.positions])
                    is_bought = ((portfolio[:, 0] == 'Stock') & (portfolio[:, 1] == figi)).sum()
                except:
                    get_portfolio = SmthWithStatus()
                    get_portfolio.status = 'Not Ok'

                # Try to get orders
                try:
                    get_orders = await aclient.get_orders(broker_account_id)
                    orders = np.array([[order.orderId, order.figi, order.operation, order.requestedLots, order.executedLots, order.price]
                                      for order in get_orders.payload])
                    if len(orders) > 0:
                        figi_ordered = (orders[:, 1] == figi).sum()
                    else:
                        figi_ordered = 0
                except:
                    get_orders = SmthWithStatus()
                    get_orders.status = 'Not Ok'

                # Checking orders and portfolio. If good -> buy
                if (is_bought == 0) & (figi_ordered == 0) & (get_portfolio.status == 'Ok') & (get_orders.status == 'Ok'):
                    try:
                        price = np.round(np.float64(cur_events[0].payload.c), 2)
                        lots = np.float64(bank_size) * 0.99 // np.float64(price)
                        body = ti.LimitOrderRequest(
                                        lots=lots,
                                        operation='Buy',
                                        price=price,
                                        )
                        await aclient.post_orders_limit_order(figi, body, broker_account_id)
                        print('Time: %s. Buy order. Price: %f. Lots: %i.' % (dt.datetime.utcnow(), price, lots))
                        order_set_time = time.time()
                        order_placed = True
                        pivot_time = round(round(pivot_time, 1) + buy_period, 1)
                        if pivot_time - time.time() < 0:
                            print('Time Error. Buy mode')
                        await asyncio.sleep(pivot_time - time.time())
                    except:
                        order_placed = False
                        print('Buy Error')
                        pivot_time = round(round(pivot_time, 1) + buy_period, 1)
                        if pivot_time - time.time() < 0:
                            print('Time Error. Buy mode')
                        await asyncio.sleep(pivot_time - time.time())
                else:
                    order_placed = False
                    pivot_time = round(round(pivot_time, 1) + buy_period, 1)
                    if pivot_time - time.time() < 0:
                        print('Time Error. Buy mode')
                    await asyncio.sleep(pivot_time - time.time())

            # Normal looping if prediction is not good
            elif ((np.round(y_pred) == 0) & (order_placed == False)) | (dt.datetime.utcnow().hour * 60 + dt.datetime.utcnow().minute in bad_minutes):
                print('Time: %s. Idle mode. Current price: %f' % (dt.datetime.utcnow(), np.round(np.float64(cur_events[0].payload.c), 2)))
                pivot_time = round(round(pivot_time, 1) + period, 1)
                if pivot_time - time.time() < 0:
                    print('Time Error. Idle mode.')
                await asyncio.sleep(pivot_time - time.time())

            # Order placed without errors. Placing take-profit and stop-loss
            elif order_placed:
                print('Time: %s. Sell mode. Current price: %f' % (dt.datetime.utcnow(), np.round(np.float64(cur_events[0].payload.c), 2)))
                order_executed = False
                tp_placed = False
                sl_placed = False
                scenario = 0
                try:
                    get_portfolio = await aclient.get_portfolio(broker_account_id)
                    portfolio = np.array([[position.instrument_type, position.figi, position.ticker, position.balance, position.lots]
                                      for position in get_portfolio.payload.positions])
                    figi_in_portfolio = (portfolio[:, 1] == figi).sum()
                except:
                    print('Error. Sell mode. Portfolio loading')
                    get_portfolio = SmthWithStatus()
                    get_portfolio.status = 'Not Ok'

                # Checking buy order state (is stock in portfolio?)
                if (get_portfolio.status == 'Ok') & (figi_in_portfolio == 1) & (scenario == 0):

                    # Checking if buy order is fully executed
                    if ((portfolio[portfolio[:, 1] == figi][0][4] == lots)):
                        scenario = 1
                        lots_to_sell = portfolio[portfolio[:, 1] == figi][0][4]

                    # Checking if time for executing buy order is up and buy order executed but not fully
                    elif ((portfolio[portfolio[:, 1] == figi][0][4] != 0) & (portfolio[portfolio[:, 1] == figi][0][4] < lots)
                          & (time.time() - order_set_time >= execute_time)):
                        scenario = 1
                        lots_to_sell = portfolio[portfolio[:, 1] == figi][0][4]

                        # Try to cancel buy order
                        try:
                            get_orders = await aclient.get_orders(broker_account_id)
                            orders = np.array([[order.orderId, order.figi, order.operation, order.requestedLots, order.executedLots, order.price]
                                              for order in get_orders.payload])
                            order_id = orders[orders[:, 1] == figi][0, 0]
                            await aclient.post_orders_cancel(order_id, broker_account_id)
                        except:
                            print('Error. Sell mode. Canceling not executed buying')
                            get_orders = SmthWithStatus()
                            get_orders.status = 'Not Ok'

                # Checking buy order state (is stock not in portfolio and execute time is up?)
                elif ((get_portfolio.status == 'Ok') & (figi_in_portfolio == 0) & (time.time() - order_set_time >= 20) & (scenario == 0)):
                    scenario = 2

                    # Try to cancel buy order because execute time is up and nothing is bought
                    try:
                        get_orders = await aclient.get_orders(broker_account_id)
                        orders = np.array([[order.orderId, order.figi, order.operation, order.requestedLots, order.executedLots, order.price]
                                          for order in get_orders.payload])
                        order_id = orders[orders[:, 1] == figi][0, 0]
                        await aclient.post_orders_cancel(order_id, broker_account_id)
                    except:
                        print('Error. Sell mode. Canceling fully not executed buying')
                        get_orders = SmthWithStatus()
                        get_orders.status = 'Not Ok'
                    order_placed = False

                # Checking if buy order is successfully executed (scenario 1)
                if scenario == 1:
                    i = 0

                    # Starting loop to execute sale
                    while ((lots_to_sell > 0) & (time.time() - order_set_time < sample_time * 3)):
                        print('Time: %s. Sell mode. Current price: %f' % (dt.datetime.utcnow(), np.round(np.float64(cur_events[0].payload.c), 2)))

                        # Checking if take-profit is reached
                        if (np.float64(cur_events[0].payload.c) >= np.round(price + tp_bound * price, 2)):
                            try:
                                # Canceling stop-loss sell order
                                try:
                                    get_orders = await aclient.get_orders(broker_account_id)
                                    orders = np.array([[order.orderId, order.figi, order.operation, order.requestedLots, order.executedLots, order.price]
                                                      for order in get_orders.payload])
                                    if len(orders) > 0:
                                        figi_ordered = (orders[:, 1] == figi).sum()
                                    else:
                                        figi_ordered = 0
                                except:
                                    print('Error. Sell Mode. Orders request before stop-loss canceling')
                                if figi_ordered > 0:
                                    try:
                                        await aclient.post_orders_cancel(orders[orders[:, 1] == figi][0, 0], broker_account_id)
                                    except:
                                        print('Error. Sell mode. Stop-loss cancel before take-profit posting')
                                tp_body = ti.LimitOrderRequest(
                                                lots=lots_to_sell,
                                                operation='Sell',
                                                price=np.round((price + tp_bound * price), 2)
                                                )
                                await aclient.post_orders_limit_order(figi, tp_body, broker_account_id)
                                print('Time: %s. Take-profit market order. Lots: %i. Summ: %f. Commission: %f.' % \
                                (dt.datetime.utcnow(), portfolio[portfolio[:, 1] == figi][0][4],
                                 np.round(np.float64(cur_events[0].payload.c), 2) * portfolio[portfolio[:, 1] == figi][0][4],
                                 np.round(np.float64(cur_events[0].payload.c) * portfolio[portfolio[:, 1] == figi][0][4] * 0.0005, 2))
                                )
                                tp_placed = True
                            except:
                                tp_placed = False

                        # Checking if stop-loss is reached
                        if (np.float64(cur_events[0].payload.c) <= np.round(price - sl_bound * price, 2)):
                            try:
                                # Canceling take-profit sell order
                                try:
                                    get_orders = await aclient.get_orders(broker_account_id)
                                    orders = np.array([[order.orderId, order.figi, order.operation, order.requestedLots, order.executedLots, order.price]
                                                      for order in get_orders.payload])
                                    if len(orders) > 0:
                                        figi_ordered = (orders[:, 1] == figi).sum()
                                    else:
                                        figi_ordered = 0
                                except:
                                    print('Error. Sell Mode. Orders request before take-profit canceling')
                                if figi_ordered > 0:
                                    try:
                                        await aclient.post_orders_cancel(orders[orders[:, 1] == figi][0, 0], broker_account_id)
                                    except:
                                        print('Error. Sell mode. Take-profit cancel before stop-loss posting')
                                sl_body = ti.LimitOrderRequest(
                                                lots=lots_to_sell,
                                                operation='Sell',
                                                price=np.round((price + tp_bound * price), 2)
                                                )
                                await aclient.post_orders_limit_order(figi, sl_body, broker_account_id)
                                print('Time: %s. Stop-loss market order. Lots: %i. Summ: %f. Commission: %f.' % \
                                (dt.datetime.utcnow(), portfolio[portfolio[:, 1] == figi][0][4],
                                 np.round(np.float64(cur_events[0].payload.c), 2) * portfolio[portfolio[:, 1] == figi][0][4],
                                 np.round(np.float64(cur_events[0].payload.c) * portfolio[portfolio[:, 1] == figi][0][4] * 0.0005, 2))
                                )
                                sl_placed = True
                            except:
                                sl_placed = False

                        # Wait until next pivot time to do another iteration
                        pivot_time = round(round(pivot_time, 1) + sell_period, 1)
                        if pivot_time - time.time() < 0:
                            print('Time Error. Sell mode 1')
                        await asyncio.sleep(pivot_time - time.time())
                        i += 1

                        # Checking if there are still lots to sell, if not -> end while loop
                        try:
                            get_portfolio = await aclient.get_portfolio(broker_account_id)
                            portfolio = np.array([[position.instrument_type, position.figi, position.ticker, position.balance, position.lots]
                                              for position in get_portfolio.payload.positions])
                            figi_in_portfolio = (portfolio[:, 1] == figi).sum()
                        except:
                            print('Error. Sell mode. Get portfolio before ending market order')
                            get_portfolio = SmthWithStatus()
                            get_portfolio.status = 'Not Ok'

                        if figi_in_portfolio == 0:
                            lots_to_sell = 0
                        else:
                            lots_to_sell = portfolio[portfolio[:, 1] == figi][0][4]

                # Checking if not default scenario is on, if so -> checking if cleaning order is needed
                if scenario != 0:
                    try:
                        get_orders = await aclient.get_orders(broker_account_id)
                        orders = np.array([[order.orderId,order.figi,order.operation,order.requestedLots,order.executedLots,order.price]
                                          for order in get_orders.payload])
                        if len(orders) > 0:
                            figi_ordered = (orders[:, 1] == figi).sum()
                        else:
                            figi_ordered = 0
                    except:
                        get_orders = SmthWithStatus()
                        get_orders.status = 'Ne Ok'
                    if figi_ordered > 0:
                        for order_id in orders[:, 0][orders[:, 1] == figi]:
                            try:
                                await aclient.post_orders_cancel(order_id, broker_account_id)
                            except:
                                print('Error. Sell mode. Order cancel after not fully sold in time')
                    try:
                        get_portfolio = await aclient.get_portfolio(broker_account_id)
                        portfolio = np.array([[position.instrument_type, position.figi, position.ticker, position.balance, position.lots]
                                          for position in get_portfolio.payload.positions])
                        figi_in_portfolio = (portfolio[:, 1] == figi).sum()
                    except:
                        print('Error. Sell mode. Portfolio loading')
                        get_portfolio = SmthWithStatus()
                        get_portfolio.status = 'Not Ok'

                    # Cleaning order
                    if figi_in_portfolio == 1:
                        try:
                            end_body = ti.LimitOrderRequest(
                                    lots=portfolio[portfolio[:, 1] == figi][0][4],
                                    operation='Sell',
                                    price=np.round(np.float64(cur_events[0].payload.c), 2)
                                    )
                            await aclient.post_orders_limit_order(figi, end_body, broker_account_id)
                            print('Time: %s. Cleaning market order. Lots: %i. Summ: %f. Commission: %f.' % \
                            (dt.datetime.utcnow(),portfolio[portfolio[:, 1] == figi][0][4],
                             np.round(np.float64(cur_events[0].payload.c), 2) * portfolio[portfolio[:, 1] == figi][0][4],
                             np.round(np.float64(cur_events[0].payload.c) * portfolio[portfolio[:, 1] == figi][0][4] * 0.0005, 2))
                            )
                        except:
                            print('Error. Sell mode. Cleaning market order posting')
                    order_placed = False

                # Waiting for next iteration
                if scenario == 0:
                    pivot_time = round(round(pivot_time, 1) + buy_period, 1)
                    if pivot_time - time.time() < 0:
                        print('Time Error. Sell mode 2')
                    await asyncio.sleep(pivot_time - time.time())
                elif scenario != 0:
                    pivot_time = pivot_times[pivot_times > (time.time() + 0.5)][0]
                    if pivot_time - time.time() < 0:
                        print('Time Error. Sell mode 2')
                    await asyncio.sleep(pivot_time - time.time())


# Preparing tasks
async def gather_tasks(token,figi,collect_duration,candle_resolution,depth,sample_time,period,broker_account_id,bank_size,lag,bound,tp_bound,sl_bound,countdown):
    tasks = [asyncio.create_task(stream(token,figi,collect_duration,candle_resolution,depth,countdown)),
             asyncio.create_task(collect(collect_duration,depth,sample_time,period,countdown)),
             asyncio.create_task(trade(token,broker_account_id,figi,bank_size,duration,sample_time,period,lag,bound,tp_bound,sl_bound,countdown))
            ]
    await asyncio.gather(*tasks)


# Run event loop
loop = asyncio.get_event_loop()
countdown = 60-time.gmtime().tm_sec-time.time()%1
loop.run_until_complete(gather_tasks(TOKEN_SANDBOX,FIGI,duration,candle_resolution,depth,sample_time,period,broker_account_id_sandbox,bank_size,lag,bound,tp_bound,sl_bound,countdown))
loop.close()
