// node --test web/app/prompt.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildMessages, buildRepairMessages, buildSchema, cleanSql, fixDoubleQuotedStrings, promptType, quoteIdent, readOnlyError,
  safeColumnName, sqliteCompat, tableNameFromFile, tidyError,
} from "./prompt.js";

test("promptType maps DuckDB types to the training vocabulary", () => {
  assert.equal(promptType("BIGINT"), "NUMBER");
  assert.equal(promptType("DOUBLE"), "NUMBER");
  assert.equal(promptType("DECIMAL(10,2)"), "NUMBER");
  assert.equal(promptType("VARCHAR"), "TEXT");
  assert.equal(promptType("DATE"), "TIME");
  assert.equal(promptType("TIMESTAMP WITH TIME ZONE"), "TIME");
  assert.equal(promptType("BOOLEAN"), "BOOLEAN");
  assert.equal(promptType("STRUCT(a INTEGER)"), "OTHERS");
});

test("identifiers are always quoted", () => {
  assert.equal(quoteIdent("Doanh thu (VND)"), '"Doanh thu (VND)"');
  assert.equal(quoteIdent('a"b'), '"a""b"');
});

test("table names from file names", () => {
  assert.equal(tableNameFromFile("Sales Report 2024.csv"), "sales_report_2024");
  assert.equal(tableNameFromFile("2024.csv"), "t_2024");
  assert.equal(tableNameFromFile("khách hàng.csv"), "khach_hang");
  assert.equal(tableNameFromFile("Đơn hàng.csv"), "don_hang");
  assert.equal(tableNameFromFile("order.csv"), "order_table");
  assert.equal(tableNameFromFile("...csv"), "data");
});

test("column names the model can write unquoted", () => {
  const used = new Set();
  assert.equal(safeColumnName("order_id", used), "order_id");
  assert.equal(safeColumnName("Name", used), "Name");
  assert.equal(safeColumnName("Doanh thu (VND)", used), "doanh_thu_vnd");
  assert.equal(safeColumnName("Tên khách hàng", used), "ten_khach_hang");
  assert.equal(safeColumnName("Order", used), "order_col");
  assert.equal(safeColumnName("2023 Sales", used), "c_2023_sales");
  assert.equal(safeColumnName("???", used), "col");
  assert.equal(safeColumnName("doanh thu vnd", used), "doanh_thu_vnd_2");
  assert.equal(safeColumnName("NAME", used), "NAME_2");
});

test("double-quoted strings become single-quoted unless they name something", () => {
  const known = ["sales", "category", "Order Date"];
  assert.equal(fixDoubleQuotedStrings('SELECT * FROM sales WHERE category = "Furniture"', known),
    "SELECT * FROM sales WHERE category = 'Furniture'");
  assert.equal(fixDoubleQuotedStrings('SELECT "category" FROM "sales" WHERE "order date" LIKE "2024-03%"', known),
    `SELECT "category" FROM "sales" WHERE "order date" LIKE '2024-03%'`);
  assert.equal(fixDoubleQuotedStrings(`SELECT * FROM t WHERE name = "O'Brien"`, known), "SELECT * FROM t WHERE name = 'O''Brien'");
  assert.equal(fixDoubleQuotedStrings(`SELECT 'say "hi"' FROM t`, known), `SELECT 'say "hi"' FROM t`);
  assert.equal(fixDoubleQuotedStrings('SELECT count(*) AS "n" FROM t ORDER BY "n"', known), 'SELECT count(*) AS "n" FROM t ORDER BY "n"');
  assert.equal(fixDoubleQuotedStrings('SELECT 1 -- "note"\nFROM t', known), 'SELECT 1 -- "note"\nFROM t');
});

test("sqliteCompat: case-insensitive LIKE, LIKE on dates and numbers, double-quoted strings", () => {
  const cols = [{ name: "order_date", type: "DATE" }, { name: "product", type: "VARCHAR" }, { name: "qty", type: "BIGINT" }];
  assert.equal(sqliteCompat(`SELECT count(*) FROM sales WHERE order_date LIKE "2024-03%"`, "sales", cols),
    "SELECT count(*) FROM sales WHERE CAST(order_date AS VARCHAR) ILIKE '2024-03%'");
  assert.equal(sqliteCompat("SELECT * FROM sales WHERE product LIKE '%laptop%' AND T1.qty NOT LIKE '1%'", "sales", cols),
    "SELECT * FROM sales WHERE product ILIKE '%laptop%' AND CAST(T1.qty AS VARCHAR) NOT ILIKE '1%'");
  assert.equal(sqliteCompat("SELECT * FROM sales WHERE product ILIKE 'a' AND product = 'LIKE me'", "sales", cols),
    "SELECT * FROM sales WHERE product ILIKE 'a' AND product = 'LIKE me'");
  assert.equal(sqliteCompat('SELECT "order_date" FROM sales WHERE "order_date" LIKE \'2024%\'', "sales", cols),
    `SELECT "order_date" FROM sales WHERE CAST("order_date" AS VARCHAR) ILIKE '2024%'`);
});

test("tidyError drops repeated context", () => {
  assert.equal(tidyError("Error: Parser Error: x\n\nLINE 1: a\n   ^\n\nLINE 1: a\n   ^"), "Parser Error: x\n\nLINE 1: a\n   ^");
});

test("schema matches the training format, with example rows as a comment", () => {
  const schema = buildSchema("sales", [{ name: "region", type: "VARCHAR" }, { name: "unit price", type: "DOUBLE" }], [["North", 9.5]]);
  assert.equal(
    schema,
    'CREATE TABLE "sales" (\n  "region" TEXT,\n  "unit price" NUMBER\n);\n\n' +
      '/* 1 example rows from "sales":\nregion | unit price\nNorth | 9.5\n*/',
  );
});

test("messages match templates.py", () => {
  const m = buildMessages("How many?", "S");
  assert.equal(m[0].role, "system");
  assert.equal(m[1].content, "### Schema:\nS\n\n### Question:\nHow many?\n\n### SQL:");
  const r = buildRepairMessages(m, "SELECT x", "no column x");
  assert.equal(r.length, 4);
  assert.equal(r[2].content, "SELECT x");
  assert.match(r[3].content, /no column x/);
});

test("cleanSql strips fences and trailing statements", () => {
  assert.equal(cleanSql("```sql\nSELECT 1;\n```"), "SELECT 1");
  assert.equal(cleanSql("SELECT 'a;b' FROM t; DROP TABLE t"), "SELECT 'a;b' FROM t");
  assert.equal(cleanSql("  SELECT count(*) FROM \"t\"  "), 'SELECT count(*) FROM "t"');
});

test("readOnlyError allows SELECT/WITH only", () => {
  assert.equal(readOnlyError("SELECT * FROM t"), null);
  assert.equal(readOnlyError("with x as (select 1) select * from x;"), null);
  assert.equal(readOnlyError("SELECT * FROM t WHERE note = 'please delete me'"), null);
  assert.equal(readOnlyError('SELECT "update_date" FROM t'), null);
  assert.match(readOnlyError("DELETE FROM t"), /SELECT/);
  assert.match(readOnlyError("SELECT 1; DROP TABLE t"), /single/);
  assert.match(readOnlyError("WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x"), /INSERT/);
  assert.match(readOnlyError("SELECT * FROM t -- \n; COPY t TO 'x'"), /single/);
  assert.match(readOnlyError(""), /empty/);
});
