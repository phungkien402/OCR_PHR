"""Configuration for OCR vital signs extraction pipeline."""

DEVICE = "cuda:0"  # fallback to "cpu" if VRAM insufficient

# Complete vital signs information with Vietnamese names, units, and ranges
VITALS_INFO = {
    "mach": {
        "label_vn": "Mạch",
        "label_en": "Pulse / Heart Rate",
        "abbrev": ["mạch", "pulse", "hr", "pul", "heart rate", "p:"],
        "unit": "lần/phút",
        "normal_range": (60, 100),
        "description": "Số lần tim đập mỗi phút",
    },
    "nhiet_do": {
        "label_vn": "Nhiệt độ",
        "label_en": "Temperature",
        "abbrev": ["nhiệt độ", "nhiệt", "temp", "t:", "°c"],
        "unit": "°C",
        "normal_range": (36.1, 37.2),
        "description": "Thân nhiệt cơ thể",
    },
    "huyet_ap": {
        "label_vn": "Huyết áp",
        "label_en": "Blood Pressure (SYS/DIA)",
        "abbrev": ["huyết áp", "ha", "bp", "sys", "dia", "blood pressure"],
        "unit": "mmHg",
        "normal_range": {"tam_thu": (90, 120), "tam_truong": (60, 80)},
        "description": "SYS = áp lực tâm thu (tim bơm), DIA = áp lực tâm trương (tim nghỉ)",
    },
    "nhip_tho": {
        "label_vn": "Nhịp thở",
        "label_en": "Respiratory Rate",
        "abbrev": ["nhịp thở", "rr", "resp", "nhịp"],
        "unit": "lần/phút",
        "normal_range": (12, 20),
        "description": "Số lần thở mỗi phút",
    },
    "can_nang": {
        "label_vn": "Cân nặng",
        "label_en": "Weight",
        "abbrev": ["cân nặng", "cân", "weight", "wt"],
        "unit": "kg",
        "normal_range": None,
        "description": "Trọng lượng cơ thể",
    },
    "chieu_cao": {
        "label_vn": "Chiều cao",
        "label_en": "Height",
        "abbrev": ["chiều cao", "cao", "height", "ht"],
        "unit": "cm",
        "normal_range": None,
        "description": "Chiều cao cơ thể",
    },
    "spo2": {
        "label_vn": "SpO2",
        "label_en": "Oxygen Saturation",
        "abbrev": ["spo2", "spо2", "o2", "oxy", "sao2"],
        "unit": "%",
        "normal_range": (95, 100),
        "description": "Nồng độ oxy trong máu",
    },
}

# Derived configs for backward compatibility
FIELD_KEYWORDS = {field: info["abbrev"] for field, info in VITALS_INFO.items()}

# Validation ranges (wider than normal — flags only clearly pathological values)
RANGES = {
    "mach":      (30, 200),
    "nhiet_do":  (34.0, 42.0),
    "huyet_ap":  {"tam_thu": (60, 250), "tam_truong": (30, 150)},
    "nhip_tho":  (5, 60),
    "can_nang":  (1, 300),
    "chieu_cao": (30, 250),
    "spo2":      (50, 100),
}
