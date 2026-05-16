"""Configuration for OCR vital signs extraction pipeline."""

DEVICE = "cuda:0"  # fallback to "cpu" if VRAM insufficient

FIELD_KEYWORDS = {
    "mach":      ["mạch", "pulse", "hr", "heart rate", "p:"],
    "nhiet_do":  ["nhiệt độ", "nhiệt", "temp", "t:", "°c"],
    "huyet_ap":  ["huyết áp", "ha", "bp", "blood pressure"],
    "nhip_tho":  ["nhịp thở", "rr", "nhịp"],
    "can_nang":  ["cân nặng", "cân", "weight", "kg"],
    "chieu_cao": ["chiều cao", "chieu cao", "chieu caa", "cao", "caa", "height", "cm"],
    "spo2":      ["spo2", "spо2", "sp02", "o2", "oxy"],
}

RANGES = {
    "mach":      (30, 200),
    "nhiet_do":  (34.0, 42.0),
    "huyet_ap":  {"tam_thu": (60, 250), "tam_truong": (30, 150)},
    "nhip_tho":  (5, 60),
    "can_nang":  (1, 300),
    "chieu_cao": (30, 250),
    "spo2":      (50, 100),
}
