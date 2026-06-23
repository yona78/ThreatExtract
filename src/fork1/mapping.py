from __future__ import annotations


SECUREBERT_TO_DNRTI: dict[str, set[str]] = {
    "APT": {"HackOrg"},
    "SECTEAM": {"SecTeam"},
    "IDTY": {"Idus", "Org"},
    "ACT": {"OffAct", "Way"},
    "OS": {"OffAct", "Way"},
    "TOOL": {"OffAct", "Way"},
    "VULID": {"Exp"},
    "VULNAME": {"Exp"},
    "MAL": {"Tool"},
    "FILE": {"SamFile"},
    "TIME": {"Time"},
    "LOC": {"Area"},
    "DOM": set(),
    "ENCR": set(),
    "IP": set(),
    "URL": set(),
    "MD5": set(),
    "PROT": set(),
    "EMAIL": set(),
    "SHA1": set(),
    "SHA2": set(),
}

CYNER_TO_DNRTI: dict[str, set[str]] = {
    "Organization": {"HackOrg", "SecTeam", "Idus", "Org"},
    "System": {"OffAct", "Way"},
    "Vulnerability": {"Exp"},
    "Malware": {"Tool"},
    "Indicator": {"SamFile"},
}

DNRTI_LABELS = {
    "HackOrg",
    "SecTeam",
    "Idus",
    "Org",
    "OffAct",
    "Way",
    "Exp",
    "Tool",
    "SamFile",
    "Time",
    "Area",
    "Purp",
    "Features",
}


def strip_bio(label: str) -> str:
    if label.startswith(("B-", "I-", "E-", "S-")):
        return label[2:]
    return label


def map_model_label_to_dnrti(model_name: str, label: str) -> set[str]:
    clean = strip_bio(label)
    normalized_model = model_name.lower()
    if normalized_model.startswith("securebert"):
        return set(SECUREBERT_TO_DNRTI.get(clean, set()))
    if normalized_model.startswith("cyner"):
        return set(CYNER_TO_DNRTI.get(clean, set()))
    raise ValueError(f"unknown model for label mapping: {model_name}")


def unique_mapped_dnrti_labels(model_name: str) -> set[str]:
    normalized_model = model_name.lower()
    if normalized_model.startswith("securebert"):
        mapping = SECUREBERT_TO_DNRTI
    elif normalized_model.startswith("cyner"):
        mapping = CYNER_TO_DNRTI
    else:
        raise ValueError(f"unknown model for label mapping: {model_name}")
    return {next(iter(labels)) for labels in mapping.values() if len(labels) == 1}
