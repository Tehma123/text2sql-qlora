// Pure helpers that help a user check a result: which chart (if any) to draw,
// a one-line plain-English reading of the SQL, and which text columns a query
// filters on (to suggest real values when it returns 0 rows).

const MAX_BARS = 30;
const MAX_POINTS = 400;
const TIME_LABEL = /^\d{4}(-\d{2}){0,2}( \d{2}:\d{2}(:\d{2})?)?$/;

/**
 * A chart only when the result is one label column and one number column
 * (e.g. revenue per region). Time-like labels (2024, 2024-03, 2024-03-01)
 * give a line in time order; anything else gives bars in the result's order.
 * result: {columns, rows (strings or null), numeric: [bool]}.
 */
export function chartSpec(result) {
  const { columns, rows, numeric } = result;
  if (columns.length !== 2 || numeric[0] === numeric[1] || rows.length < 2) return null;
  const v = numeric[0] ? 0 : 1;
  const l = 1 - v;
  const points = rows.filter((r) => r[v] !== null && r[l] !== null).map((r) => ({ label: r[l], value: Number(r[v]) }));
  if (points.length < 2 || points.some((p) => !Number.isFinite(p.value))) return null;
  const isTime = points.every((p) => TIME_LABEL.test(p.label));
  if (isTime) {
    if (points.length > MAX_POINTS) return null;
    points.sort((a, b) => (a.label < b.label ? -1 : a.label > b.label ? 1 : 0));
    return { kind: "line", labelName: columns[l], valueName: columns[v], points };
  }
  if (points.length > MAX_BARS) return null;
  return { kind: "bar", labelName: columns[l], valueName: columns[v], points };
}

// ---------- SQL reading ----------

/** Splits on `sep` at parenthesis depth 0, outside quotes. */
function splitTop(text, sep) {
  const parts = [];
  let depth = 0, quote = null, start = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) { if (ch === quote) quote = null; continue; }
    if (ch === "'" || ch === '"') quote = ch;
    else if (ch === "(") depth++;
    else if (ch === ")") depth--;
    else if (depth === 0 && text.startsWith(sep, i)) { parts.push(text.slice(start, i)); start = i + sep.length; }
  }
  parts.push(text.slice(start));
  return parts.map((p) => p.trim()).filter(Boolean);
}

const unquote = (s) => s.trim().replace(/^"(.*)"$/, "$1").replace(/^\w+\./, "");
const AGG = { sum: "the total", avg: "the average", max: "the highest", min: "the lowest" };

function describeExpr(expr) {
  const e = expr.trim().replace(/\s+as\s+("?[\w ]+"?)$/i, "");
  let m;
  if (/^count\s*\(\s*\*\s*\)$/i.test(e)) return "the number of rows";
  if ((m = e.match(/^count\s*\(\s*distinct\s+(.+)\)$/i))) return `the number of different ${unquote(m[1])} values`;
  if ((m = e.match(/^count\s*\((.+)\)$/i))) return `the number of rows with a ${unquote(m[1])}`;
  if ((m = e.match(/^(sum|avg|max|min)\s*\((.+)\)$/i))) {
    const inner = describeExpr(m[2]);
    return inner ? `${AGG[m[1].toLowerCase()]} ${inner.replace(/^the /, "")}` : null;
  }
  if ((m = e.match(/^strftime\s*\(\s*'%Y-%m'\s*,\s*(.+)\)$/i))) return `the month of ${unquote(m[1])}`;
  if ((m = e.match(/^strftime\s*\(\s*'%Y'\s*,\s*(.+)\)$/i))) return `the year of ${unquote(m[1])}`;
  if ((m = e.match(/^cast\s*\((.+?)\s+as\s+varchar\s*\)$/i))) return unquote(m[1]);
  if (/^[\w"]+(\.[\w"]+)?$/.test(e)) return unquote(e);
  return null;
}

/** Result column name for chart axes: "sum(revenue)" -> "total revenue"; unknown forms unchanged. */
export function readableName(expr) {
  return describeExpr(expr)?.replace(/^the /, "") ?? expr;
}

function describeCondition(cond) {
  const c = cond.trim();
  let m;
  if ((m = c.match(/^(.+?)\s+(not\s+)?i?like\s+'([^']*)'$/i))) {
    const col = describeExpr(m[1]);
    const not = m[2] ? "does not " : "";
    const pat = m[3];
    if (!col) return null;
    if (/^%[^%_]+%$/.test(pat)) return `${col} ${not ? "does not contain" : "contains"} “${pat.slice(1, -1)}”`;
    if (/^[^%_]+%$/.test(pat)) return `${col} ${not}${not ? "start" : "starts"} with “${pat.slice(0, -1)}”`;
    if (/^%[^%_]+$/.test(pat)) return `${col} ${not}${not ? "end" : "ends"} with “${pat.slice(1)}”`;
    return `${col} ${not ? "does not match" : "matches"} “${pat}”`;
  }
  if ((m = c.match(/^(.+?)\s*(=|!=|<>|>=|<=|>|<)\s*('(?:[^']|'')*'|-?[\d.]+)$/))) {
    const col = describeExpr(m[1]);
    if (!col) return null;
    const ops = { "=": "is", "!=": "is not", "<>": "is not", ">": "is more than", "<": "is less than", ">=": "is at least", "<=": "is at most" };
    const value = m[3].startsWith("'") ? `“${m[3].slice(1, -1).replaceAll("''", "'")}”` : m[3];
    return `${col} ${ops[m[2]]} ${value}`;
  }
  if ((m = c.match(/^(.+?)\s+between\s+(.+?)\s+and\s+(.+)$/i))) {
    const col = describeExpr(m[1]);
    return col ? `${col} is between ${m[2].replaceAll("'", "")} and ${m[3].replaceAll("'", "")}` : null;
  }
  return null;
}

/**
 * One sentence describing a simple single-table query, e.g. "Shows the total
 * revenue for each region, from highest to lowest, keeping the first 3.".
 * Returns null for anything it cannot read reliably (joins, subqueries,
 * unions, window functions), rather than guessing.
 */
export function explainSql(sql) {
  const s = sql.trim().replace(/;$/, "").replace(/\s+/g, " ");
  if (/\b(join|union|intersect|except|over|with)\b/i.test(s) || /\(\s*select\b/i.test(s)) return null;
  const m = s.match(/^select\s+(distinct\s+)?(.+?)\s+from\s+("?[\w]+"?)(?:\s+(?:as\s+)?\w+)?(?:\s+where\s+(.+?))?(?:\s+group\s+by\s+(.+?))?(?:\s+having\s+(.+?))?(?:\s+order\s+by\s+(.+?))?(?:\s+limit\s+(\d+))?$/i);
  if (!m) return null;
  const [, distinct, selectList, , where, groupBy, having, orderBy, limit] = m;
  if (having) return null;

  const groups = groupBy ? splitTop(groupBy, ",").map((g) => describeExpr(g)?.replace(/^the /, "") ?? null) : [];
  if (groups.includes(null)) return null;
  const selectExprs = selectList === "*" ? [] : splitTop(selectList, ",");
  let items = selectList === "*" ? ["every column"] : selectExprs.map(describeExpr);
  if (items.includes(null)) return null;
  items = items.filter((it) => !groups.includes(it.replace(/^the /, "")));
  const onlyGroups = !items.length && groups.length > 0; // e.g. SELECT region ... GROUP BY region
  if (onlyGroups) items = groups.map((g) => `each ${g}`);
  const list = items.length > 1 ? `${items.slice(0, -1).join(", ")} and ${items.at(-1)}` : items[0];

  let text = `Shows ${distinct ? "the different values of " : ""}${list}`;
  if (groups.length && !onlyGroups) text += ` for each ${groups.join(" and ")}`;
  if (where) {
    const conds = splitTop(where.replace(/\s+and\s+/gi, " AND "), " AND ").map(describeCondition);
    if (conds.includes(null) || /\bor\b/i.test(where)) text += `, only for rows where ${where}`;
    else text += `, only for rows where ${conds.join(" and ")}`;
  }
  if (orderBy) {
    const keys = splitTop(orderBy, ",");
    const first = keys[0].match(/^(.+?)(?:\s+(asc|desc))?$/i);
    const key = /^\d+$/.test(first[1]) ? selectExprs[Number(first[1]) - 1] : first[1]; // ORDER BY 2 = 2nd column
    const what = key && describeExpr(key)?.replace(/^the /, "");
    if (!what) return null;
    const desc = (first[2] ?? "").toLowerCase() === "desc";
    text += desc ? `, from highest to lowest ${what}` : `, from lowest to highest ${what}`;
  }
  if (limit) text += limit === "1" ? (orderBy ? ", keeping only the top one" : ", keeping one row") : `, keeping the first ${limit}`;
  return `${text}.`;
}

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const identifier = (name) => `(?:"${escapeRegex(name)}"|\\b${escapeRegex(name)}\\b)`;

/**
 * `text_column = 'literal'` (or != / <>) comparisons in the query, so the app
 * can check the literal against the data: [{column, literal}] with the
 * literal unescaped. Used to fix case ('Japan' when the data says 'japan').
 */
export function textEqualities(sql, columns) {
  const found = [];
  for (const c of columns.filter((col) => /^(VARCHAR|TEXT|STRING|CHAR)/i.test(col.type))) {
    const re = new RegExp(`${identifier(c.name)}\\s*(?:=|!=|<>)\\s*'((?:[^']|'')*)'`, "gi");
    for (const m of sql.matchAll(re)) {
      const literal = m[1].replaceAll("''", "'");
      if (!found.some((f) => f.column === c.name && f.literal === literal)) found.push({ column: c.name, literal });
    }
  }
  return found;
}

/** Replaces `column <op> 'from'` with `column <op> 'to'` (only in such comparisons). */
export function replaceTextLiteral(sql, column, from, to) {
  const re = new RegExp(`(${identifier(column)}\\s*(?:=|!=|<>)\\s*)'${escapeRegex(from.replaceAll("'", "''"))}'`, "gi");
  return sql.replace(re, (_, head) => `${head}'${to.replaceAll("'", "''")}'`);
}

/** Number columns compared with a number (= 1970, > 5): their real range helps when nothing matches. */
export function filteredNumberColumns(sql, columns) {
  return columns.filter((c) => /^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|U|FLOAT|REAL|DOUBLE|DECIMAL)/i.test(c.type))
    .filter((c) => new RegExp(`${identifier(c.name)}\\s*(?:=|!=|<>|>=|<=|>|<)\\s*-?\\d`, "i").test(sql))
    .map((c) => c.name);
}

/** True when a result has no rows, or is one row of only NULL / 0 (an aggregate over nothing). */
export function looksEmpty(result) {
  if (result.total === 0) return true;
  return result.total === 1 && result.rows[0].every((v) => v === null || v === "0");
}

/** Text columns compared with a string literal in the query (= 'x', LIKE 'x', IN ('x', ...)). */
export function filteredTextColumns(sql, columns) {
  const text = columns.filter((c) => /^(VARCHAR|TEXT|STRING|CHAR)/i.test(c.type)).map((c) => c.name);
  const found = new Set();
  for (const name of text) {
    const id = `(?:"${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"|\\b${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b)`;
    const re = new RegExp(`${id}\\s*(?:=|!=|<>|(?:not\\s+)?i?like|(?:not\\s+)?in\\s*\\()\\s*'`, "i");
    if (re.test(sql)) found.add(name);
  }
  return [...found];
}
