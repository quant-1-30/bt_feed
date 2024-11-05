
calendar_pipeline = [
    dict(type="CsvLoader"),
    dict(type="Date2Int"),
    dict(type="OrmWriter", alias="postgres", table="calendar")
]

instrument_pipeline = [
    # DatasetPath
    dict(type="CsvLoader"),
    dict(type="Duplicate"),
    dict(type="Date2Int"),
    dict(type="Sliced", fields=["sid", "name", "first_trading", "delist"]),
    dict(type="OrmWriter", alias="postgres", table="instrument")
]

ticker_pipeline = [
    dict(type="Restricted", alias="stock"),
    dict(type="StructLoader", alias="struct"),
    dict(type="Normalize", alias="normalize"),
    dict(type="UTC", alias="utc"),
    dict(type="ProcessNa"),
    dict(type="ProcessInf"),
    dict(type="Mulitply", alias="multiply"),
    dict(type="Sliced", fields=["sid", "open", "high", "low", "close", "volume", "amount", "utc"]),
    dict(type="OrmWriter",alias="postgres", table="minute")
]

adjustment_pipeline = [
    dict(type="CsvLoader"),
    dict(type="Normalize", alias="normalize"),
    dict(type="UTC", alias="utc"),
    dict(type="ProcessNa"),
    dict(type="ProcessInf"),
    dict(type="Mulitply", alias="multiply"),
    dict(type="Mulitply", alias="multiply", fields=["share", "transfer", "interest"]),
    dict(type="Sliced", fields=["sid", "register_date", "ex_date", "share", "transfer", "interest"]),
    dict(type="OrmWriter",alias="postgres", table="adjustment")
]

rights_pipeline = [
    dict(type="CsvLoader"),
    dict(type="Normalize", alias="normalize"),
    dict(type="UTC", alias="utc"),
    dict(type="ProcessNa"),
    dict(type="ProcessInf"),
    dict(type="Mulitply", alias="multiply"),
    dict(type="Mulitply", alias="multiply", fields=["price", "ratio"]),
    dict(type="Sliced", fields=["sid", "register_date", "ex_date", "effective_date", "price", "ratio"]),
    dict(type="OrmWriter",alias="postgres", table="rights")
]
