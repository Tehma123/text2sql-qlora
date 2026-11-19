// Pure helpers shared by the app and its Node tests: schema text, chat
// messages, SQL clean-up and the read-only check. Must match the training
// format in src/text2sql/prompts/templates.py.

export const SYSTEM_PROMPT =
  "You are a SQL expert. Given a database schema and a question, write the " +
  "SQLite query that answers the question.\n" +
  "Respond with the SQL query only -- no explanation, no markdown code fences.";

export const MAX_REPAIRS = 2;

/** DuckDB column type -> the type vocabulary of the training schemas. */
export function promptType(duckdbType) {
  const t = duckdbType.toUpperCase();
  if (/^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|FLOAT|REAL|DOUBLE|DECIMAL)/.test(t)) return "NUMBER";
  if (/^(VARCHAR|TEXT|STRING|CHAR)/.test(t)) return "TEXT";
  if (/^(DATE|TIME|TIMESTAMP|INTERVAL)/.test(t)) return "TIME";
  if (t === "BOOLEAN") return "BOOLEAN";
  return "OTHERS";
}

/** SQL identifier in double quotes, with embedded quotes doubled. */
export function quoteIdent(name) {
  return `"${String(name).replaceAll('"', '""')}"`;
}

// Words the model would have to quote if used as a bare column name. The
// model was trained on Spider, where every name is a plain identifier, so it
// writes names unquoted; it cannot be relied on to quote them.
const RESERVED = new Set(("all and any as asc between by case cast column create cross current date default delete desc distinct " +
  "drop else end except exists false fetch for from full group having in inner insert intersect interval into is join " +
  "key left like limit natural not null offset on or order outer over partition primary references right select set " +
  "table then time timestamp to true union unique update user using values when where window with").split(" "));

/** Lower-case ASCII snake_case: strips accents (đ -> d) and turns everything else into "_". */
function asciiSnake(text) {
  return text.normalize("NFD").replace(/\p{M}/gu, "").replace(/[đĐ]/g, "d").toLowerCase()
    .replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

/** Table name from a file name: ASCII snake_case, never empty or digit-first. */
export function tableNameFromFile(fileName) {
  let base = asciiSnake(fileName.replace(/\.[^.]*$/, ""));
  if (!base) base = "data";
  if (/^[0-9]/.test(base)) base = `t_${base}`;
  if (RESERVED.has(base)) base = `${base}_table`;
  return base;
}

/**
 * Column name the model can write without quotes. Plain identifiers are kept
 * as they are; anything else becomes ASCII snake_case ("Doanh thu (VND)" ->
 * "doanh_thu_vnd"). `used` collects names already taken (lower-cased), so the
 * result is unique within the table.
 */
export function safeColumnName(name, used) {
  let base = String(name);
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(base) || RESERVED.has(base.toLowerCase())) {
    base = asciiSnake(base) || "col";
    if (/^[0-9]/.test(base)) base = `c_${base}`;
    if (RESERVED.has(base)) base = `${base}_col`;
  }
  let candidate = base;
  for (let i = 2; used.has(candidate.toLowerCase()); i++) candidate = `${base}_${i}`;
  used.add(candidate.toLowerCase());
  return candidate;
}

/** DuckDB repeats the "LINE 1: ..." context in some errors; keep each paragraph once. */
export function tidyError(message) {
  const seen = new Set();
  return String(message).replace(/^Error:\s*/, "").split(/\n\n/)
    .filter((part) => !seen.has(part) && seen.add(part)).join("\n\n").trim();
}

function sampleValue(v) {
  if (v === null || v === undefined) return "NULL";
  const s = String(v);
  return s.length > 40 ? `${s.slice(0, 37)}...` : s;
}

/**
 * CREATE TABLE text in the training format, followed by a few example rows as
 * an SQL comment so the model sees real value spellings (e.g. 'North' vs 'N').
 * columns: [{name, type}] with DuckDB types; sampleRows: arrays in column order.
 */
export function buildSchema(table, columns, sampleRows = []) {
  const cols = columns.map((c) => `  ${quoteIdent(c.name)} ${promptType(c.type)}`).join(",\n");
  let text = `CREATE TABLE ${quoteIdent(table)} (\n${cols}\n);`;
  if (sampleRows.length) {
    const header = columns.map((c) => c.name).join(" | ");
    const lines = sampleRows.map((r) => r.map(sampleValue).join(" | "));
    text += `\n\n/* ${sampleRows.length} example rows from ${quoteIdent(table)}:\n${header}\n${lines.join("\n")}\n*/`;
  }
  return text;
}

export function buildMessages(question, schema) {
  return [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: `### Schema:\n${schema}\n\n### Question:\n${question}\n\n### SQL:` },
  ];
}

/** Follow-up turn asking the model to fix a query that failed. */
export function buildRepairMessages(messages, badSql, error) {
  return [
    ...messages,
    { role: "assistant", content: badSql },
    {
      role: "user",
      content: `That query failed with this error:\n${error}\n\nWrite a corrected SQLite query for the same question.\n\n### SQL:`,
    },
  ];
}

/** Model output -> one SQL statement: strips code fences and anything after the first statement. */
export function cleanSql(text) {
  let s = String(text).trim();
  const fence = s.match(/```(?:sql)?\s*([\s\S]*?)(?:```|$)/i);
  if (fence) s = fence[1].trim();
  const end = statementEnd(s);
  if (end !== -1) s = s.slice(0, end);
  return s.trim();
}

/**
 * SQLite reads "x" as a string when no column is called x, and the model
 * (trained on Spider's SQLite queries) writes strings that way:
 * WHERE category = "Furniture". DuckDB always reads "x" as a name. This turns
 * every double-quoted token that is not a known table, column or AS alias
 * into a single-quoted string, which is what SQLite would have run.
 */
export function fixDoubleQuotedStrings(sql, knownNames) {
  const known = new Set([...knownNames].map((n) => String(n).toLowerCase()));
  for (const m of sql.matchAll(/\bas\s+"((?:[^"]|"")*)"/gi)) known.add(m[1].replaceAll('""', '"').toLowerCase());
  let out = "";
  for (let i = 0; i < sql.length; ) {
    const ch = sql[i];
    if (ch === "'" || ch === '"') {
      let j = i + 1;
      while (j < sql.length && !(sql[j] === ch && sql[j + 1] !== ch)) j += sql[j] === ch ? 2 : 1;
      const token = sql.slice(i, j + 1);
      if (ch === '"') {
        const content = token.slice(1, -1).replaceAll('""', '"');
        out += known.has(content.toLowerCase()) ? token : `'${content.replaceAll("'", "''")}'`;
      } else out += token;
      i = j + 1;
    } else if (sql.startsWith("--", i)) {
      const j = sql.indexOf("\n", i);
      out += sql.slice(i, j === -1 ? sql.length : j);
      i = j === -1 ? sql.length : j;
    } else {
      out += ch;
      i++;
    }
  }
  return out;
}

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Makes SQLite-style SQL behave the same on DuckDB:
 *  - "x" strings become 'x' (see fixDoubleQuotedStrings);
 *  - LIKE becomes ILIKE, since SQLite's LIKE ignores case and DuckDB's does not;
 *  - LIKE on a non-text column (dates, numbers) compares its text form, as in
 *    SQLite, where order_date LIKE '2024-03%' works because dates are text.
 * columns: [{name, type}] of the loaded table (DuckDB types).
 */
export function sqliteCompat(sql, table, columns) {
  const fixed = fixDoubleQuotedStrings(sql, [table, ...columns.map((c) => c.name)]);
  const nonText = columns.filter((c) => promptType(c.type) !== "TEXT").map((c) => c.name);
  const castLike = nonText.map((name) => new RegExp(
    `((?:\\b\\w+\\.)?(?:"${escapeRegex(name)}"|\\b${escapeRegex(name)}\\b))(\\s+(?:NOT\\s+)?ILIKE\\b)`, "gi"));
  // Rewrite only outside single-quoted literals (odd split parts are literals).
  return fixed.split(/('(?:[^']|'')*')/).map((part, i) => {
    if (i % 2) return part;
    let s = part.replace(/\bLIKE\b/gi, (m, offset, whole) => (/I$/i.test(whole.slice(0, offset)) ? m : "ILIKE"));
    for (const re of castLike) s = s.replace(re, "CAST($1 AS VARCHAR)$2");
    return s;
  }).join("");
}

/** Index of the first `;` outside string literals and quoted identifiers, or -1. */
function statementEnd(s) {
  let quote = null;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (quote) {
      if (ch === quote) {
        if (s[i + 1] === quote) i++;
        else quote = null;
      }
    } else if (ch === "'" || ch === '"') quote = ch;
    else if (ch === ";") return i;
  }
  return -1;
}

function stripLiteralsAndComments(s) {
  return s
    .replace(/--[^\n]*/g, " ")
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/'(?:[^']|'')*'/g, "''")
    .replace(/"(?:[^"]|"")*"/g, '""');
}

const FORBIDDEN = /\b(insert|update|delete|merge|drop|alter|create|replace|truncate|copy|attach|detach|install|load|pragma|set|reset|export|import|call|vacuum|checkpoint)\b/i;

/**
 * Read-only check before running: one statement, starting with SELECT or
 * WITH, with no write/DDL/config keyword outside literals. Returns an error
 * message, or null when the query may run.
 */
export function readOnlyError(sql) {
  const bare = stripLiteralsAndComments(sql).trim();
  if (!bare) return "The query is empty.";
  const rest = bare.replace(/;\s*$/, "");
  if (rest.includes(";")) return "Only a single SQL statement can be run.";
  if (!/^(select|with)\b/i.test(rest)) return "Only SELECT queries can be run (the query must start with SELECT or WITH).";
  const bad = rest.match(FORBIDDEN);
  if (bad) return `Only read-only queries can be run ("${bad[1].toUpperCase()}" is not allowed).`;
  return null;
}
