from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import paths


def generate_us_holidays(years: list[int]) -> pd.DataFrame:
    try:
        import holidays
    except ImportError:
        return _generate_us_federal_holidays_with_pandas(years)

    calendar = holidays.US(years=years)
    rows = [
        {"date": pd.to_datetime(date).date(), "holiday_name": name, "is_holiday": True}
        for date, name in sorted(calendar.items())
    ]
    return pd.DataFrame(rows, columns=["date", "holiday_name", "is_holiday"])


def _generate_us_federal_holidays_with_pandas(years: list[int]) -> pd.DataFrame:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    calendar = USFederalHolidayCalendar()
    start = f"{min(years)}-01-01"
    end = f"{max(years)}-12-31"
    rows = []
    for rule in calendar.rules:
        dates = rule.dates(start, end)
        for date in dates:
            if date.year in years:
                rows.append(
                    {
                        "date": date.date(),
                        "holiday_name": rule.name,
                        "is_holiday": True,
                    }
                )
    return (
        pd.DataFrame(rows, columns=["date", "holiday_name", "is_holiday"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def write_us_holidays(years: list[int], output_path: Path | None = None) -> Path:
    if output_path is None:
        label = "_".join(str(year) for year in years)
        output_path = paths.external_data / f"us_holidays_{label}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_us_holidays(years).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate US holiday CSV files.")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = write_us_holidays(args.years, args.output)
    print(output)


if __name__ == "__main__":
    main()
