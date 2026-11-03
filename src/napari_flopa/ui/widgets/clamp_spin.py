from qtpy.QtGui import QValidator
from qtpy.QtWidgets import QDoubleSpinBox, QSpinBox


class _ClampMixin:
    """Accept any typed number and clamp it to the range on Enter / focus-out.

    A plain spin box refuses keystrokes that leave the range, or snaps back to
    the previous value. Here a number outside the range can be typed, and it
    becomes the nearest bound.
    """

    def _parse(self, text: str) -> float | None:
        body = text.strip()
        prefix, suffix = self.prefix(), self.suffix()
        if prefix and body.startswith(prefix):
            body = body[len(prefix) :]
        if suffix and body.endswith(suffix):
            body = body[: -len(suffix)]
        body = body.strip().replace(self.locale().groupSeparator(), "")
        body = body.replace(self.locale().decimalPoint(), ".")
        try:
            return float(body)
        except ValueError:
            return None

    def validate(self, text: str, pos: int):
        state, text, pos = super().validate(text, pos)
        if state == QValidator.Invalid and self._parse(text) is not None:
            state = QValidator.Intermediate
        return state, text, pos

    def fixup(self, text: str) -> str:
        value = self._parse(text)
        if value is None:
            return super().fixup(text)
        value = min(max(value, self.minimum()), self.maximum())
        if isinstance(self, QSpinBox):
            value = int(round(value))
        return self.textFromValue(value)


class ClampDoubleSpinBox(_ClampMixin, QDoubleSpinBox):
    """QDoubleSpinBox that clamps out-of-range input instead of refusing it."""


class ClampSpinBox(_ClampMixin, QSpinBox):
    """QSpinBox that clamps out-of-range input instead of refusing it."""
