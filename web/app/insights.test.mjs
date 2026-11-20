// node --test web/app/insights.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  chartSpec, explainSql, filteredNumberColumns, filteredTextColumns, looksEmpty, readableName, replaceTextLiteral, textEqualities,
} from "./insights.js";

test("chartSpec: bars for label + number, in result order", () => {
  const spec = chartSpec({ columns: ["sum(revenue)", "region"], rows: [["45050", "South"], ["37503.5", "North"]], numeric: [true, false] });
  assert.equal(spec.kind, "bar");
  assert.equal(spec.labelName, "region");
  assert.deepEqual(spec.points, [{ label: "South", value: 45050 }, { label: "North", value: 37503.5 }]);
});

test("chartSpec: line for time labels, sorted by time", () => {
  const spec = chartSpec({ columns: ["month", "total"], rows: [["2024-02", "5"], ["2024-01", "3"]], numeric: [false, true] });
  assert.equal(spec.kind, "line");
  assert.deepEqual(spec.points.map((p) => p.label), ["2024-01", "2024-02"]);
});

test("chartSpec: no chart for one row, two numbers, three columns or too many bars", () => {
  assert.equal(chartSpec({ columns: ["n"], rows: [["1"]], numeric: [true] }), null);
  assert.equal(chartSpec({ columns: ["a", "b"], rows: [["1", "2"], ["3", "4"]], numeric: [true, true] }), null);
  assert.equal(chartSpec({ columns: ["a", "b"], rows: [["x", "1"]], numeric: [false, true] }), null);
  const many = Array.from({ length: 31 }, (_, i) => [`p${i}`, String(i)]);
  assert.equal(chartSpec({ columns: ["p", "n"], rows: many, numeric: [false, true] }), null);
});

test("explainSql reads simple single-table queries", () => {
  assert.equal(explainSql("SELECT sum(revenue) ,  region FROM sales GROUP BY region"), "Shows the total revenue for each region.");
  assert.equal(explainSql("SELECT product FROM sales GROUP BY product ORDER BY sum(quantity) DESC LIMIT 3"),
    "Shows each product, from highest to lowest total quantity, keeping the first 3.");
  assert.equal(explainSql("SELECT count(*) FROM sales WHERE customer_type  =  'New'"),
    "Shows the number of rows, only for rows where customer_type is “New”.");
  assert.equal(explainSql("SELECT count(*) FROM sales WHERE CAST(order_date AS VARCHAR) ILIKE '2024-03%'"),
    "Shows the number of rows, only for rows where order_date starts with “2024-03”.");
  assert.equal(explainSql("SELECT region FROM sales WHERE category = 'Furniture' GROUP BY region ORDER BY sum(revenue) DESC LIMIT 1"),
    "Shows each region, only for rows where category is “Furniture”, from highest to lowest total revenue, keeping only the top one.");
  assert.equal(explainSql("SELECT count(DISTINCT product) FROM sales WHERE region = 'West'"),
    "Shows the number of different product values, only for rows where region is “West”.");
});

test("explainSql and readableName handle month grouping", () => {
  assert.equal(explainSql("SELECT sum(revenue), strftime('%Y-%m', order_date) FROM sales GROUP BY strftime('%Y-%m', order_date)"),
    "Shows the total revenue for each month of order_date.");
  assert.equal(explainSql("SELECT region, sum(revenue) FROM sales GROUP BY region ORDER BY 2 DESC"),
    "Shows the total revenue for each region, from highest to lowest total revenue.");
  assert.equal(explainSql("SELECT region FROM sales ORDER BY 5"), null);
  assert.equal(explainSql("SELECT avg(tip/total_bill) FROM tips"), null); // was a crash: avg() of an expression
  assert.equal(readableName("avg(tip/total_bill)"), "avg(tip/total_bill)");
  assert.equal(readableName("sum(revenue)"), "total revenue");
  assert.equal(readableName("strftime('%Y-%m', order_date)"), "month of order_date");
  assert.equal(readableName("count(*)"), "number of rows");
  assert.equal(readableName("revenue * 2"), "revenue * 2");
});

test("explainSql gives up on joins, subqueries and HAVING", () => {
  assert.equal(explainSql("SELECT a FROM t JOIN u ON t.id = u.id"), null);
  assert.equal(explainSql("SELECT count(*) * 100 / (SELECT count(*) FROM t) FROM t"), null);
  assert.equal(explainSql("SELECT a FROM t GROUP BY a HAVING count(*) > 2"), null);
});

test("text equalities, literal replacement, number filters, empty-looking results", () => {
  const cols = [{ name: "origin", type: "VARCHAR" }, { name: "model_year", type: "BIGINT" }, { name: "name", type: "VARCHAR" }];
  assert.deepEqual(textEqualities("SELECT count(*) FROM mpg WHERE origin = 'Japan' AND name <> 'O''Neil'", cols),
    [{ column: "origin", literal: "Japan" }, { column: "name", literal: "O'Neil" }]);
  assert.equal(replaceTextLiteral("SELECT * FROM mpg WHERE origin  =  'Japan' OR origin != 'Japan'", "origin", "Japan", "japan"),
    "SELECT * FROM mpg WHERE origin  =  'japan' OR origin != 'japan'");
  assert.equal(replaceTextLiteral("SELECT 'Japan' AS x FROM mpg WHERE name = 'Japan'", "origin", "Japan", "japan"),
    "SELECT 'Japan' AS x FROM mpg WHERE name = 'Japan'");
  assert.deepEqual(filteredNumberColumns("SELECT avg(weight) FROM mpg WHERE model_year = 1970", cols), ["model_year"]);
  assert.equal(looksEmpty({ total: 0, rows: [] }), true);
  assert.equal(looksEmpty({ total: 1, rows: [["0"]] }), true);
  assert.equal(looksEmpty({ total: 1, rows: [[null]] }), true);
  assert.equal(looksEmpty({ total: 1, rows: [["79"]] }), false);
});

test("filteredTextColumns finds text columns compared with literals", () => {
  const cols = [{ name: "product", type: "VARCHAR" }, { name: "category", type: "VARCHAR" }, { name: "qty", type: "BIGINT" }];
  assert.deepEqual(filteredTextColumns("SELECT * FROM t WHERE product = 'Furniture' AND qty = 3", cols), ["product"]);
  assert.deepEqual(filteredTextColumns("SELECT * FROM t WHERE category ILIKE '%x%' OR \"product\" IN ('a')", cols), ["product", "category"]);
  assert.deepEqual(filteredTextColumns("SELECT product FROM t", cols), []);
});
