from clearml.utilities.version import Version


def test_local_version_segment_does_not_crash():
    # PEP 440 local versions such as PyTorch's "2.1.0+cu121" are valid and must
    # be parseable. _cmpkey's local handling used to raise IndexError on any
    # single-segment local version, even though is_valid_version_string accepts
    # it.
    for spec in ("2.1.0+cu121", "1.0+cpu", "2.0.1+cu118"):
        assert Version.is_valid_version_string(spec)
        assert Version(spec).local == spec.split("+", 1)[1]


def test_local_version_ordering():
    # Absent local sorts before a present one; within a local segment,
    # alphanumeric parts sort before numeric parts (PEP 440).
    assert Version("2.1.0+cu118") < Version("2.1.0+cu121")
    assert Version("1.0") < Version("1.0+abc")
    assert Version("1.0+abc") < Version("1.0+1")
