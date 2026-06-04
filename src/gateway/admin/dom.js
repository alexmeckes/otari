export const $ = (id) => document.getElementById(id);

export function selectedValue(id) {
  const element = $(id);
  return element ? element.value : "";
}
