
from unittest import TestCase

from lexnlp.extract.common.annotations.money_annotation import MoneyAnnotation
from lexnlp.extract.en.money import get_money, get_money_annotations
from lexnlp.tests.typed_annotations_tests import TypedAnnotationsTester


class TestMoneyPlain(TestCase):

    def test_money(self):
        text = "100 bucks, 100 dollars, 100 greens"
        ds = list(get_money(text))
        self.assertEqual(1, len(ds))

        ants = list(get_money_annotations(text))
        self.assertEqual(1, len(ds))
        self.assertEqual('en', ants[0].locale)
        self.assertEqual('USD', ants[0].currency)
        self.assertEqual(100.0, ants[0].amount)

    def test_file_samples(self):
        tester = TypedAnnotationsTester()
        tester.test_and_raise_errors(
            get_money_annotations_sorted,
            'lexnlp/typed_annotations/en/money/money.txt',
            MoneyAnnotation)


def get_money_annotations_sorted(text):
    ants = list(get_money_annotations(text))
    ants.sort(key=lambda a: a.coords[0])
    return ants
