import { formatSales } from "../utils/format.js";

// Dropdown over all stores returned by the API. There are 1,115 of them, so
// each option carries enough context (type, assortment, average sales) to make
// a choice meaningful rather than picking a number at random.
export default function StoreSelector({ stores, selectedStoreId, onChange, disabled }) {
  return (
    <div className="control">
      <label className="control-label" htmlFor="store-select">
        Store
      </label>
      <select
        id="store-select"
        className="control-input"
        value={selectedStoreId ?? ""}
        disabled={disabled || stores.length === 0}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {stores.map((store) => (
          <option key={store.store_id} value={store.store_id}>
            Store {store.store_id} · type {store.store_type} · assortment{" "}
            {store.assortment} · avg {formatSales(store.average_daily_sales)}
          </option>
        ))}
      </select>
    </div>
  );
}
