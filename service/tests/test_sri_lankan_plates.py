"""
Comprehensive tests for Sri Lankan license plate validation and formatting.
This is pure logic — no mocking or external dependencies needed.
"""

import pytest
from utils.sri_lankan_plates import (
    normalize_text,
    validate_modern_format,
    validate_provincial_numeric,
    validate_old_format,
    validate_special_format,
    validate_sri_lankan_plate,
    smart_format_plate,
    correct_ocr_errors,
    is_valid_plate,
    get_province_name,
    SL_PROVINCE_CODES,
)


class TestNormalizeText:
    def test_uppercase(self):
        assert normalize_text("abc") == "ABC"

    def test_remove_spaces(self):
        assert normalize_text("WP CA 1234") == "WPCA1234"

    def test_remove_dashes(self):
        assert normalize_text("WP CA-1234") == "WPCA1234"

    def test_mixed(self):
        assert normalize_text("wp ca-1234") == "WPCA1234"


class TestModernFormat:
    def test_valid_two_letter(self):
        valid, formatted = validate_modern_format("WP CA-1234")
        assert valid is True
        assert formatted == "WP CA-1234"

    def test_valid_three_letter(self):
        valid, formatted = validate_modern_format("WP CAB-1234")
        assert valid is True
        assert formatted == "WP CAB-1234"

    def test_valid_no_separator(self):
        valid, formatted = validate_modern_format("WPCA1234")
        assert valid is True
        assert formatted == "WP CA-1234"

    def test_all_provinces(self):
        for code in SL_PROVINCE_CODES:
            valid, formatted = validate_modern_format(f"{code}AB1234")
            assert valid is True, f"Province {code} should be valid"

    def test_invalid_province(self):
        valid, _ = validate_modern_format("XX AB-1234")
        assert valid is False

    def test_too_few_digits(self):
        valid, _ = validate_modern_format("WP CA-123")
        assert valid is False

    def test_too_many_digits(self):
        valid, _ = validate_modern_format("WP CA-12345")
        assert valid is False


class TestProvincialNumeric:
    def test_valid(self):
        valid, formatted = validate_provincial_numeric("WP 1234")
        assert valid is True
        assert formatted == "WP 1234"

    def test_valid_no_space(self):
        valid, formatted = validate_provincial_numeric("WP1234")
        assert valid is True
        assert formatted == "WP 1234"

    def test_invalid_province(self):
        valid, _ = validate_provincial_numeric("XX 1234")
        assert valid is False

    def test_too_few_digits(self):
        valid, _ = validate_provincial_numeric("WP 123")
        assert valid is False


class TestOldFormat:
    def test_valid_two_digit_prefix(self):
        valid, formatted = validate_old_format("12-3456")
        assert valid is True
        assert formatted == "12-3456"

    def test_valid_three_digit_prefix(self):
        valid, formatted = validate_old_format("123-4567")
        assert valid is True
        assert formatted == "123-4567"

    def test_valid_no_separator(self):
        valid, formatted = validate_old_format("123456")
        assert valid is True

    def test_too_short(self):
        valid, _ = validate_old_format("1234")
        assert valid is False


class TestSpecialFormat:
    def test_valid(self):
        valid, formatted = validate_special_format("CAR 1234")
        assert valid is True
        assert formatted == "CAR 1234"

    def test_valid_no_space(self):
        valid, formatted = validate_special_format("GOV1234")
        assert valid is True
        assert formatted == "GOV 1234"

    def test_two_letters(self):
        valid, _ = validate_special_format("CA 1234")
        assert valid is False


class TestValidateSriLankanPlate:
    def test_empty_string(self):
        valid, _, plate_type = validate_sri_lankan_plate("")
        assert valid is False
        assert plate_type == "empty"

    def test_modern(self):
        valid, formatted, plate_type = validate_sri_lankan_plate("WP CAB-1234")
        assert valid is True
        assert plate_type == "modern"

    def test_provincial(self):
        valid, formatted, plate_type = validate_sri_lankan_plate("CP 5678")
        assert valid is True
        assert plate_type == "provincial"

    def test_old(self):
        valid, formatted, plate_type = validate_sri_lankan_plate("12-3456")
        assert valid is True
        assert plate_type == "old"

    def test_special(self):
        valid, formatted, plate_type = validate_sri_lankan_plate("CAR 1234")
        assert valid is True
        assert plate_type == "special"

    def test_invalid(self):
        valid, _, plate_type = validate_sri_lankan_plate("INVALID")
        assert valid is False
        assert plate_type == "unknown"


class TestSmartFormatPlate:
    def test_empty(self):
        result, confidence = smart_format_plate("")
        assert result is None
        assert confidence == 0.0

    def test_valid_plate(self):
        result, confidence = smart_format_plate("WP CAB-1234")
        assert result == "WP CAB-1234"
        assert confidence == 1.0

    def test_too_short(self):
        result, confidence = smart_format_plate("AB")
        assert result is None
        assert confidence == 0.0

    def test_noisy_input(self):
        result, confidence = smart_format_plate("WP-CA.1234")
        assert result is not None
        assert confidence > 0.0


class TestOCRCorrection:
    def test_correct_o_to_zero(self):
        expected = {0: "digit"}
        result = correct_ocr_errors("O", expected)
        assert result == "0"

    def test_correct_i_to_one(self):
        expected = {0: "digit"}
        result = correct_ocr_errors("I", expected)
        assert result == "1"

    def test_correct_zero_to_o(self):
        expected = {0: "letter"}
        result = correct_ocr_errors("0", expected)
        assert result == "O"

    def test_no_correction_needed(self):
        expected = {0: "letter"}
        result = correct_ocr_errors("A", expected)
        assert result == "A"

    def test_position_out_of_range(self):
        expected = {5: "digit"}
        result = correct_ocr_errors("AB", expected)
        assert result == "AB"


class TestUtilityFunctions:
    def test_is_valid_plate_true(self):
        assert is_valid_plate("WP CA-1234") is True

    def test_is_valid_plate_false(self):
        assert is_valid_plate("INVALID") is False

    def test_get_province_name_valid(self):
        assert get_province_name("WP CA-1234") == "Western Province"

    def test_get_province_name_invalid(self):
        assert get_province_name("XX CA-1234") is None

    def test_get_province_name_short(self):
        assert get_province_name("W") is None

    def test_all_province_codes(self):
        for code, name in SL_PROVINCE_CODES.items():
            assert get_province_name(f"{code} 1234") == name
