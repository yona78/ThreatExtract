from fork1.mapping import unique_mapped_dnrti_labels


def test_unique_mapped_labels_exclude_one_to_many_securebert_labels() -> None:
    labels = unique_mapped_dnrti_labels("securebert")

    assert {"HackOrg", "SecTeam", "Exp", "Tool", "SamFile", "Time", "Area"} <= labels
    assert "Way" not in labels
    assert "OffAct" not in labels
    assert "Org" not in labels


def test_unique_mapped_labels_exclude_one_to_many_cyner_labels() -> None:
    labels = unique_mapped_dnrti_labels("cyner")

    assert labels == {"Exp", "Tool", "SamFile"}
