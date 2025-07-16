from common.log_parser import LogParser

if __name__ == '__main__':
    parser = LogParser(['/nas-1/ShareFolder/bb/prod_logs/binance3/binance3_20241231031158.log'], analyze_latency=True, exchange='binance')
    df_order, df_panel = parser.get_order_and_panel()

    print('\n[external] lws receive exchange send ack (uds) - lws write send')
    stat = ((df_order['send_ack_time_logger'] - df_order['send_time_lws']).dt.total_seconds() * 1000).describe(percentiles=[0.5, 0.9, 0.99])
    print(stat)

    print('\n[external] lws receive exchange order end msg (uds) - lws write cancel')
    stat = ((df_order['order_end_time_logger'] - df_order['cancel_time_lws']).dt.total_seconds() * 1000).describe(percentiles=[0.5, 0.9, 0.99])
    print(stat)

    # help check internel latency

    # print('\n[internel] lws write order - EC write order')
    # stat = ((df_order['send_time_lws'] - df_order['send_time_logger']).dt.total_seconds() * 1000).describe(percentiles=[0.5, 0.9, 0.99])
    # print(stat)

    # print('\n[internel] lws write order - EC trigger market data timestamp')
    # stat = ((df_order['send_time_lws'] - df_order['send_time']).dt.total_seconds() * 1000).describe(percentiles=[0.5, 0.9, 0.99])
    # print(stat)

    # print('\n[internel] lws write order - EC write order')
    # stat = ((df_order['cancel_time_lws'] - df_order['cancel_time_logger']).dt.total_seconds() * 1000).describe(percentiles=[0.5, 0.9, 0.99])
    # print(stat)

    # print('\n[internel] lws write order - EC trigger market data timestamp')
    # stat = ((df_order['cancel_time_lws'] - df_order['cancel_time']).dt.total_seconds() * 1000).describe(percentiles=[0.5, 0.9, 0.99])
    # print(stat)
