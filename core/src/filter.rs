#![allow(dead_code)]

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Bool(bool),
}

impl FilterValue {
    fn display_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Number(n) => format_number(*n),
            Self::Bool(b) => b.to_string(),
        }
    }

    const fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }
}

fn format_number(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{n:.0}")
    } else {
        n.to_string()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterOp {
    Eq(FilterValue),
    Ne(FilterValue),
    In(Vec<FilterValue>),
    Nin(Vec<FilterValue>),
    Gt(FilterValue),
    Gte(FilterValue),
    Lt(FilterValue),
    Lte(FilterValue),
    Contains(String),
    Icontains(String),
    Wildcard,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpr {
    Field(String, FilterOp),
    And(Vec<Self>),
    Or(Vec<Self>),
    Not(Box<Self>),
}

#[allow(clippy::float_cmp)]
fn values_equal(value: &FilterValue, payload_str: &str) -> bool {
    if let (Some(n), Ok(p)) = (value.as_number(), payload_str.parse::<f64>()) {
        return n == p;
    }
    value.display_string() == payload_str
}

fn numeric_order(value: &FilterValue, payload_str: &str) -> Option<std::cmp::Ordering> {
    if let (Some(n), Ok(p)) = (value.as_number(), payload_str.parse::<f64>()) {
        return p.partial_cmp(&n);
    }
    Some(payload_str.cmp(&value.display_string()))
}

fn evaluate_op(op: &FilterOp, field_value: Option<&str>) -> bool {
    if matches!(op, FilterOp::Wildcard) {
        return true;
    }
    let Some(payload_str) = field_value else {
        return false;
    };
    match op {
        FilterOp::Wildcard => true,
        FilterOp::Eq(v) => values_equal(v, payload_str),
        FilterOp::Ne(v) => !values_equal(v, payload_str),
        FilterOp::In(values) => values.iter().any(|v| values_equal(v, payload_str)),
        FilterOp::Nin(values) => !values.iter().any(|v| values_equal(v, payload_str)),
        FilterOp::Gt(v) => numeric_order(v, payload_str) == Some(std::cmp::Ordering::Greater),
        FilterOp::Gte(v) => matches!(numeric_order(v, payload_str), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)),
        FilterOp::Lt(v) => numeric_order(v, payload_str) == Some(std::cmp::Ordering::Less),
        FilterOp::Lte(v) => matches!(numeric_order(v, payload_str), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)),
        FilterOp::Contains(needle) => payload_str.contains(needle.as_str()),
        FilterOp::Icontains(needle) => payload_str.to_lowercase().contains(&needle.to_lowercase()),
    }
}

#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn evaluate(expr: &FilterExpr, payload: &HashMap<String, String>) -> bool {
    match expr {
        FilterExpr::Field(field, op) => evaluate_op(op, payload.get(field).map(String::as_str)),
        FilterExpr::And(exprs) => exprs.iter().all(|e| evaluate(e, payload)),
        FilterExpr::Or(exprs) => exprs.iter().any(|e| evaluate(e, payload)),
        FilterExpr::Not(inner) => !evaluate(inner, payload),
    }
}

fn parse_filter_value(value: &serde_json::Value) -> Result<FilterValue, String> {
    match value {
        serde_json::Value::String(s) => Ok(FilterValue::String(s.clone())),
        serde_json::Value::Number(n) => n.as_f64().map(FilterValue::Number).ok_or_else(|| format!("filter number {n} is out of range")),
        serde_json::Value::Bool(b) => Ok(FilterValue::Bool(*b)),
        other => Err(format!("filter value {other} must be a string, number, or boolean")),
    }
}

fn parse_filter_values(value: &serde_json::Value) -> Result<Vec<FilterValue>, String> {
    let serde_json::Value::Array(items) = value else {
        return Err(format!("filter value {value} must be an array"));
    };
    items.iter().map(parse_filter_value).collect()
}

fn parse_field_condition(field: &str, condition: &serde_json::Value) -> Result<Vec<FilterExpr>, String> {
    match condition {
        serde_json::Value::Object(ops) => {
            let mut exprs = Vec::with_capacity(ops.len());
            for (op_name, op_value) in ops {
                let op = match op_name.as_str() {
                    "eq" => FilterOp::Eq(parse_filter_value(op_value)?),
                    "ne" => FilterOp::Ne(parse_filter_value(op_value)?),
                    "in" => FilterOp::In(parse_filter_values(op_value)?),
                    "nin" => FilterOp::Nin(parse_filter_values(op_value)?),
                    "gt" => FilterOp::Gt(parse_filter_value(op_value)?),
                    "gte" => FilterOp::Gte(parse_filter_value(op_value)?),
                    "lt" => FilterOp::Lt(parse_filter_value(op_value)?),
                    "lte" => FilterOp::Lte(parse_filter_value(op_value)?),
                    "contains" => FilterOp::Contains(expect_string(op_value)?),
                    "icontains" => FilterOp::Icontains(expect_string(op_value)?),
                    other => return Err(format!("unknown filter operator {other:?}")),
                };
                exprs.push(FilterExpr::Field(field.to_string(), op));
            }
            Ok(exprs)
        }
        serde_json::Value::String(s) if s == "*" => Ok(vec![FilterExpr::Field(field.to_string(), FilterOp::Wildcard)]),
        _ => Ok(vec![FilterExpr::Field(field.to_string(), FilterOp::Eq(parse_filter_value(condition)?))]),
    }
}

fn expect_string(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        other => Err(format!("filter value {other} must be a string")),
    }
}

#[must_use = "the parsed filter must be used"]
#[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
pub fn parse_filter_expr(value: &serde_json::Value) -> Result<FilterExpr, String> {
    let serde_json::Value::Object(map) = value else {
        return Err(format!("filters must be a JSON object, got {value}"));
    };
    let mut clauses = Vec::new();
    for (key, condition) in map {
        match key.as_str() {
            "AND" => clauses.push(FilterExpr::And(parse_filter_list(condition)?)),
            "OR" => clauses.push(FilterExpr::Or(parse_filter_list(condition)?)),
            "NOT" => {
                let inner = parse_filter_list(condition)?;
                clauses.push(FilterExpr::Not(Box::new(FilterExpr::And(inner))));
            }
            field => clauses.extend(parse_field_condition(field, condition)?),
        }
    }
    Ok(if clauses.len() == 1 { clauses.into_iter().next().expect("length checked") } else { FilterExpr::And(clauses) })
}

fn parse_filter_list(value: &serde_json::Value) -> Result<Vec<FilterExpr>, String> {
    let serde_json::Value::Array(items) = value else {
        return Err(format!("AND/OR/NOT must hold a JSON array, got {value}"));
    };
    items.iter().map(parse_filter_expr).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    fn parse(json: &str) -> FilterExpr {
        let value: serde_json::Value = serde_json::from_str(json).expect("test JSON must be valid");
        parse_filter_expr(&value).expect("test filter must parse")
    }

    #[test]
    fn parses_a_bare_equality_condition() {
        let expr = parse(r#"{"user_id": "alice"}"#);
        assert_eq!(expr, FilterExpr::Field("user_id".to_string(), FilterOp::Eq(FilterValue::String("alice".to_string()))));
    }

    #[test]
    fn parses_an_eq_operator_condition() {
        let expr = parse(r#"{"user_id": {"eq": "alice"}}"#);
        assert_eq!(expr, FilterExpr::Field("user_id".to_string(), FilterOp::Eq(FilterValue::String("alice".to_string()))));
    }

    #[test]
    fn parses_multiple_operators_on_the_same_field_as_an_implicit_and() {
        let expr = parse(r#"{"score": {"gte": 5, "lte": 10}}"#);
        let FilterExpr::And(clauses) = expr else { panic!("expected an implicit AND") };
        assert_eq!(clauses.len(), 2);
    }

    #[test]
    fn parses_multiple_top_level_fields_as_an_implicit_and() {
        let expr = parse(r#"{"user_id": "alice", "memory_type": "fact"}"#);
        let FilterExpr::And(clauses) = expr else { panic!("expected an implicit AND") };
        assert_eq!(clauses.len(), 2);
    }

    #[test]
    fn parses_an_explicit_and() {
        let expr = parse(r#"{"AND": [{"category": "programming"}, {"priority": {"gte": 5}}]}"#);
        assert!(matches!(expr, FilterExpr::And(clauses) if clauses.len() == 2));
    }

    #[test]
    fn parses_an_explicit_or() {
        let expr = parse(r#"{"OR": [{"category": "programming"}, {"category": "design"}]}"#);
        assert!(matches!(expr, FilterExpr::Or(clauses) if clauses.len() == 2));
    }

    #[test]
    fn parses_a_not() {
        let expr = parse(r#"{"NOT": [{"category": "spam"}]}"#);
        assert!(matches!(expr, FilterExpr::Not(_)));
    }

    #[test]
    fn parses_an_in_and_nin_list() {
        let in_expr = parse(r#"{"category": {"in": ["a", "b"]}}"#);
        assert_eq!(
            in_expr,
            FilterExpr::Field(
                "category".to_string(),
                FilterOp::In(vec![FilterValue::String("a".to_string()), FilterValue::String("b".to_string())])
            )
        );
        let nin_expr = parse(r#"{"category": {"nin": ["a", "b"]}}"#);
        assert!(matches!(nin_expr, FilterExpr::Field(_, FilterOp::Nin(_))));
    }

    #[test]
    fn parses_a_wildcard_string_into_the_wildcard_op() {
        let expr = parse(r#"{"user_id": "*"}"#);
        assert_eq!(expr, FilterExpr::Field("user_id".to_string(), FilterOp::Wildcard));
    }

    #[test]
    fn rejects_an_unknown_operator() {
        let value: serde_json::Value = serde_json::from_str(r#"{"field": {"bogus": 1}}"#).expect("valid JSON");
        assert!(parse_filter_expr(&value).is_err());
    }

    #[test]
    fn rejects_a_non_object_top_level_value() {
        let value: serde_json::Value = serde_json::from_str(r#""not an object""#).expect("valid JSON");
        assert!(parse_filter_expr(&value).is_err());
    }

    #[test]
    fn eq_matches_a_real_payload_field() {
        let expr = FilterExpr::Field("user_id".to_string(), FilterOp::Eq(FilterValue::String("alice".to_string())));
        assert!(evaluate(&expr, &payload(&[("user_id", "alice")])));
        assert!(!evaluate(&expr, &payload(&[("user_id", "bob")])));
    }

    #[test]
    fn eq_is_false_when_the_field_is_absent() {
        let expr = FilterExpr::Field("user_id".to_string(), FilterOp::Eq(FilterValue::String("alice".to_string())));
        assert!(!evaluate(&expr, &payload(&[])));
    }

    #[test]
    fn ne_is_the_real_inverse_of_eq() {
        let expr = FilterExpr::Field("user_id".to_string(), FilterOp::Ne(FilterValue::String("alice".to_string())));
        assert!(!evaluate(&expr, &payload(&[("user_id", "alice")])));
        assert!(evaluate(&expr, &payload(&[("user_id", "bob")])));
    }

    #[test]
    fn in_matches_any_real_listed_value() {
        let expr = FilterExpr::Field(
            "category".to_string(),
            FilterOp::In(vec![FilterValue::String("a".to_string()), FilterValue::String("b".to_string())]),
        );
        assert!(evaluate(&expr, &payload(&[("category", "a")])));
        assert!(evaluate(&expr, &payload(&[("category", "b")])));
        assert!(!evaluate(&expr, &payload(&[("category", "c")])));
    }

    #[test]
    fn nin_is_the_real_inverse_of_in() {
        let expr = FilterExpr::Field(
            "category".to_string(),
            FilterOp::Nin(vec![FilterValue::String("a".to_string()), FilterValue::String("b".to_string())]),
        );
        assert!(!evaluate(&expr, &payload(&[("category", "a")])));
        assert!(evaluate(&expr, &payload(&[("category", "c")])));
    }

    #[test]
    fn wildcard_matches_regardless_of_value_or_absence() {
        let expr = FilterExpr::Field("user_id".to_string(), FilterOp::Wildcard);
        assert!(evaluate(&expr, &payload(&[("user_id", "anyone")])));
        assert!(evaluate(&expr, &payload(&[])), "wildcard must match even when the field is entirely absent");
    }

    #[test]
    fn eq_matches_a_number_against_its_real_stored_string_form() {
        let expr = FilterExpr::Field("priority".to_string(), FilterOp::Eq(FilterValue::Number(10.0)));
        assert!(evaluate(&expr, &payload(&[("priority", "10")])));
        assert!(!evaluate(&expr, &payload(&[("priority", "11")])));
    }

    #[test]
    fn gt_gte_lt_lte_compare_numerically_when_both_sides_are_real_numbers() {
        let field = HashMap::from([("priority".to_string(), "10".to_string())]);
        assert!(evaluate(&FilterExpr::Field("priority".to_string(), FilterOp::Gt(FilterValue::Number(5.0))), &field));
        assert!(!evaluate(&FilterExpr::Field("priority".to_string(), FilterOp::Gt(FilterValue::Number(10.0))), &field));
        assert!(evaluate(&FilterExpr::Field("priority".to_string(), FilterOp::Gte(FilterValue::Number(10.0))), &field));
        assert!(evaluate(&FilterExpr::Field("priority".to_string(), FilterOp::Lt(FilterValue::Number(20.0))), &field));
        assert!(!evaluate(&FilterExpr::Field("priority".to_string(), FilterOp::Lt(FilterValue::Number(10.0))), &field));
        assert!(evaluate(&FilterExpr::Field("priority".to_string(), FilterOp::Lte(FilterValue::Number(10.0))), &field));
    }

    #[test]
    fn gt_gte_lt_lte_fall_back_to_lexicographic_comparison_on_a_real_expiration_date_shaped_field() {
        let field = HashMap::from([("expiration_date".to_string(), "2026-06-15".to_string())]);
        assert!(evaluate(
            &FilterExpr::Field("expiration_date".to_string(), FilterOp::Gt(FilterValue::String("2026-01-01".to_string()))),
            &field
        ));
        assert!(!evaluate(
            &FilterExpr::Field("expiration_date".to_string(), FilterOp::Lt(FilterValue::String("2026-01-01".to_string()))),
            &field
        ));
        assert!(evaluate(
            &FilterExpr::Field("expiration_date".to_string(), FilterOp::Lte(FilterValue::String("2026-06-15".to_string()))),
            &field
        ));
    }

    #[test]
    fn gt_is_false_when_the_field_is_absent() {
        let expr = FilterExpr::Field("priority".to_string(), FilterOp::Gt(FilterValue::Number(5.0)));
        assert!(!evaluate(&expr, &payload(&[])));
    }

    #[test]
    fn contains_matches_a_real_substring() {
        let expr = FilterExpr::Field("content".to_string(), FilterOp::Contains("engineer".to_string()));
        assert!(evaluate(&expr, &payload(&[("content", "Alice is an engineer.")])));
        assert!(!evaluate(&expr, &payload(&[("content", "Alice is a designer.")])));
    }

    #[test]
    fn contains_is_case_sensitive() {
        let expr = FilterExpr::Field("content".to_string(), FilterOp::Contains("Engineer".to_string()));
        assert!(!evaluate(&expr, &payload(&[("content", "alice is an engineer.")])));
    }

    #[test]
    fn icontains_matches_regardless_of_case() {
        let expr = FilterExpr::Field("content".to_string(), FilterOp::Icontains("ENGINEER".to_string()));
        assert!(
            evaluate(&expr, &payload(&[("content", "Alice is an engineer.")])),
            "icontains must be genuinely case-insensitive (ADR-30)"
        );
    }

    #[test]
    fn icontains_still_fails_on_a_real_non_match() {
        let expr = FilterExpr::Field("content".to_string(), FilterOp::Icontains("designer".to_string()));
        assert!(!evaluate(&expr, &payload(&[("content", "Alice is an engineer.")])));
    }

    #[test]
    fn parses_range_and_substring_operators_from_real_json() {
        assert_eq!(parse(r#"{"priority": {"gt": 5}}"#), FilterExpr::Field("priority".to_string(), FilterOp::Gt(FilterValue::Number(5.0))));
        assert_eq!(
            parse(r#"{"content": {"contains": "engineer"}}"#),
            FilterExpr::Field("content".to_string(), FilterOp::Contains("engineer".to_string()))
        );
        assert_eq!(
            parse(r#"{"content": {"icontains": "ENGINEER"}}"#),
            FilterExpr::Field("content".to_string(), FilterOp::Icontains("ENGINEER".to_string()))
        );
    }

    #[test]
    fn and_requires_every_real_clause_to_hold() {
        let expr = FilterExpr::And(vec![
            FilterExpr::Field("category".to_string(), FilterOp::Eq(FilterValue::String("engineering".to_string()))),
            FilterExpr::Field("priority".to_string(), FilterOp::Gte(FilterValue::Number(5.0))),
        ]);
        assert!(evaluate(&expr, &payload(&[("category", "engineering"), ("priority", "8")])));
        assert!(!evaluate(&expr, &payload(&[("category", "engineering"), ("priority", "2")])));
        assert!(!evaluate(&expr, &payload(&[("category", "sales"), ("priority", "8")])));
    }

    #[test]
    fn or_requires_only_one_real_clause_to_hold() {
        let expr = FilterExpr::Or(vec![
            FilterExpr::Field("category".to_string(), FilterOp::Eq(FilterValue::String("engineering".to_string()))),
            FilterExpr::Field("category".to_string(), FilterOp::Eq(FilterValue::String("design".to_string()))),
        ]);
        assert!(evaluate(&expr, &payload(&[("category", "engineering")])));
        assert!(evaluate(&expr, &payload(&[("category", "design")])));
        assert!(!evaluate(&expr, &payload(&[("category", "sales")])));
    }

    #[test]
    fn not_inverts_a_real_inner_expression() {
        let expr = FilterExpr::Not(Box::new(FilterExpr::Field(
            "category".to_string(),
            FilterOp::Eq(FilterValue::String("spam".to_string())),
        )));
        assert!(evaluate(&expr, &payload(&[("category", "engineering")])));
        assert!(!evaluate(&expr, &payload(&[("category", "spam")])));
    }

    #[test]
    fn and_of_or_of_not_evaluates_correctly_nested_one_level_deeper() {
        let category_is_engineering_or_design = FilterExpr::Or(vec![
            FilterExpr::Field("category".to_string(), FilterOp::Eq(FilterValue::String("engineering".to_string()))),
            FilterExpr::Field("category".to_string(), FilterOp::Eq(FilterValue::String("design".to_string()))),
        ]);
        let priority_is_not_low = FilterExpr::Not(Box::new(FilterExpr::Field(
            "priority".to_string(),
            FilterOp::Lt(FilterValue::Number(5.0)),
        )));
        let expr = FilterExpr::And(vec![category_is_engineering_or_design, priority_is_not_low]);
        assert!(evaluate(&expr, &payload(&[("category", "engineering"), ("priority", "8")])), "engineering, high priority must match");
        assert!(evaluate(&expr, &payload(&[("category", "design"), ("priority", "5")])), "design, priority at the boundary must match");
        assert!(!evaluate(&expr, &payload(&[("category", "engineering"), ("priority", "2")])), "engineering but low priority must not match");
        assert!(!evaluate(&expr, &payload(&[("category", "sales"), ("priority", "8")])), "wrong category must not match regardless of priority");
    }

    #[test]
    fn a_real_compound_filter_from_parsed_json_evaluates_correctly_end_to_end() {
        let expr = parse(r#"{"AND": [{"category": {"in": ["engineering", "design"]}}, {"priority": {"gte": 5}}]}"#);
        assert!(evaluate(&expr, &payload(&[("category", "engineering"), ("priority", "8")])));
        assert!(!evaluate(&expr, &payload(&[("category", "sales"), ("priority", "8")])));
        assert!(!evaluate(&expr, &payload(&[("category", "engineering"), ("priority", "2")])));
    }
}
