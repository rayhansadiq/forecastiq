// Single place for surfacing loading and error state, so no component has to
// invent its own. Error text comes from the API's `detail` field wherever the
// backend supplied one.
export default function StatusBanner({ kind, message }) {
  if (!message) return null;
  return (
    <div className={`banner banner-${kind}`} role={kind === "error" ? "alert" : "status"}>
      {message}
    </div>
  );
}
