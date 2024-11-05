from configs.pipeline.tdx import calendar_pipeline, instrument_pipeline, ticker_pipeline, adjustment_pipeline, rights_pipeline

# dataset settings
calendar_dataset = dict(
    workers=4,
    root_path = "~/Downloads/quant/calendar",
    path_module = {"type": "CalendarPath"},
    pipeline=calendar_pipeline,
)

instrument_dataset = dict(
    workers=4,
    root_path = "~/Downloads/quant/assets",
    path_module = {"type": "InstrumentPath"},
    pipeline=instrument_pipeline,
)

ticker_dataset = dict(
    workers=4,
    root_path = "~/Downloads/quant/lines",
    path_module = {"type": "DatasetPath", "inner_dir": "minute"},
    pipeline=ticker_pipeline,
)

adjustment_dataset = dict(
    workers=4,
    root_path = "~/Downloads/quant/adjustements",
    path_module = {"type": "AdjustmentPath"},
    pipeline=adjustment_pipeline,
)

right_dataset = dict(
    workers=4,
    root_path = "~/Downloads/quant/rights",
    path_module = {"type": "RightsPath"},
    pipeline=rights_pipeline,
)