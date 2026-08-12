// Shared formatting helpers, so number and date rendering is consistent
// across every component rather than reinvented in each one.

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

const decimalFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function formatSales(value) {
  if (value === null || value === undefined) return "-";
  return numberFormatter.format(value);
}

export function formatDecimal(value) {
  if (value === null || value === undefined) return "-";
  return decimalFormatter.format(value);
}

export function formatPercent(value) {
  if (value === null || value === undefined) return "-";
  return `${decimalFormatter.format(value)}%`;
}

// "2015-08-01" -> "Aug 1". Parsed manually rather than with new Date(string)
// to avoid the browser shifting the date by a timezone offset.
export function formatShortDate(isoDate) {
  const [year, month, day] = isoDate.split("-").map(Number);
  const monthNames = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${monthNames[month - 1]} ${day}`;
}

export function formatLongDate(isoDate) {
  const [year] = isoDate.split("-").map(Number);
  return `${formatShortDate(isoDate)}, ${year}`;
}

const WEEKDAY_NAMES = {
  1: "Monday",
  2: "Tuesday",
  3: "Wednesday",
  4: "Thursday",
  5: "Friday",
  6: "Saturday",
  7: "Sunday",
};

export function weekdayName(dayOfWeek) {
  return WEEKDAY_NAMES[dayOfWeek] || "";
}
