#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityType {
    Proper,
    Quoted,
}

fn is_capitalized_word(word: &str) -> bool {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) if first.is_uppercase() => chars.all(char::is_alphabetic),
        _ => false,
    }
}

fn extract_proper_spans(text: &str) -> Vec<(EntityType, String)> {
    let mut spans = Vec::new();
    let mut run: Vec<&str> = Vec::new();
    for token in text.split_whitespace() {
        let ends_sentence = token.ends_with(['.', '!', '?']);
        let trimmed = token.trim_matches(|c: char| !c.is_alphanumeric());
        if !trimmed.is_empty() && is_capitalized_word(trimmed) {
            run.push(trimmed);
            if ends_sentence {
                if run.len() > 1 {
                    spans.push((EntityType::Proper, run.join(" ")));
                }
                run.clear();
            }
        } else {
            if run.len() > 1 {
                spans.push((EntityType::Proper, run.join(" ")));
            }
            run.clear();
        }
    }
    if run.len() > 1 {
        spans.push((EntityType::Proper, run.join(" ")));
    }
    spans
}

fn extract_quoted_spans(text: &str) -> Vec<(EntityType, String)> {
    let mut spans = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('"') {
        let after_open = &rest[start + 1..];
        if let Some(end) = after_open.find('"') {
            let quoted = after_open[..end].trim();
            if quoted.len() > 2 {
                spans.push((EntityType::Quoted, quoted.to_string()));
            }
            rest = &after_open[end + 1..];
        } else {
            break;
        }
    }
    spans
}

const fn is_single_quote_open_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, '(' | '[' | '{' | ',' | ';')
}

const fn is_single_quote_close_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']')
}

fn extract_single_quoted_spans(text: &str) -> Vec<(EntityType, String)> {
    let mut spans = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        let opens = chars[i].1 == '\'' && (i == 0 || is_single_quote_open_boundary(chars[i - 1].1));
        if !opens {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < len && chars[j].1 != '\'' {
            j += 1;
        }
        let closes = j < len && (j + 1 == len || is_single_quote_close_boundary(chars[j + 1].1));
        if !closes {
            i += 1;
            continue;
        }
        let inner = text.get(chars[i + 1].0..chars[j].0).unwrap_or("").trim();
        if inner.len() > 2 {
            spans.push((EntityType::Quoted, inner.to_string()));
        }
        i = j + 1;
    }
    spans
}

#[must_use]
pub fn extract_entities(text: &str) -> Vec<(EntityType, String)> {
    let mut entities = extract_proper_spans(text);
    entities.extend(extract_quoted_spans(text));
    entities.extend(extract_single_quoted_spans(text));
    entities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_entities_finds_a_real_multi_word_proper_span() {
        let entities = extract_entities("Alice met Bob Smith at the conference.");
        assert!(entities.contains(&(EntityType::Proper, "Bob Smith".to_string())));
    }

    #[test]
    fn extract_entities_does_not_extract_a_single_capitalized_word() {
        let entities = extract_entities("The weather was nice today.");
        assert!(
            !entities.iter().any(|(t, _)| *t == EntityType::Proper),
            "a lone capitalized word, especially sentence-initial, is too noisy a signal without real POS context"
        );
    }

    #[test]
    fn extract_entities_treats_a_sentence_boundary_as_two_separate_spans() {
        let entities = extract_entities("New York is large. Golden Gate is a bridge.");
        assert!(entities.contains(&(EntityType::Proper, "New York".to_string())));
        assert!(entities.contains(&(EntityType::Proper, "Golden Gate".to_string())));
        assert!(
            !entities.iter().any(|(_, text)| text.contains("York is large") || text.contains("York")
                && text.contains("Golden")),
            "a sentence boundary must not merge two unrelated proper spans into one"
        );
    }

    #[test]
    fn extract_entities_does_not_merge_two_proper_spans_directly_adjacent_across_a_sentence_boundary() {
        let entities = extract_entities("New York. Golden Gate is a bridge.");
        assert!(entities.contains(&(EntityType::Proper, "New York".to_string())));
        assert!(entities.contains(&(EntityType::Proper, "Golden Gate".to_string())));
        assert!(
            !entities.contains(&(EntityType::Proper, "New York Golden Gate".to_string())),
            "a sentence-ending period must break a proper-noun run even with no lowercase word between the two spans"
        );
    }

    #[test]
    fn extract_entities_finds_a_real_quoted_phrase() {
        let entities = extract_entities(r#"She said "hello world" to everyone."#);
        assert!(entities.contains(&(EntityType::Quoted, "hello world".to_string())));
    }

    #[test]
    fn extract_entities_rejects_a_too_short_quoted_fragment() {
        let entities = extract_entities(r#"He said "ok" and left."#);
        assert!(
            !entities.iter().any(|(t, _)| *t == EntityType::Quoted),
            "a quoted fragment of 2 chars or fewer must be rejected, the documented minimum-length rule"
        );
    }

    #[test]
    fn extract_entities_does_not_misparse_an_apostrophe_as_a_quote_boundary() {
        let entities = extract_entities("Alice's report was thorough.");
        assert!(
            !entities.iter().any(|(t, _)| *t == EntityType::Quoted),
            "an apostrophe in a contraction/possessive must never be treated as a quote boundary"
        );
    }

    #[test]
    fn extract_entities_returns_nothing_for_plain_lowercase_text() {
        let entities = extract_entities("this is just a plain sentence with nothing special.");
        assert!(entities.is_empty());
    }

    #[test]
    fn extract_entities_finds_a_real_single_quoted_phrase() {
        let entities = extract_entities("the term 'quantum entanglement' appears in the paper.");
        assert!(entities.contains(&(EntityType::Quoted, "quantum entanglement".to_string())));
    }

    #[test]
    fn extract_entities_does_not_extract_a_contraction_as_a_single_quoted_span() {
        let entities = extract_entities("I don't think that's right.");
        assert!(
            !entities.iter().any(|(t, _)| *t == EntityType::Quoted),
            "the apostrophe in a contraction is preceded by a letter, not a boundary, and must not be treated as a quote"
        );
    }

    #[test]
    fn extract_entities_does_not_extract_a_possessive_as_a_single_quoted_span() {
        let entities = extract_entities("Bob's book was on the table.");
        assert!(
            !entities.iter().any(|(t, _)| *t == EntityType::Quoted),
            "the apostrophe in a possessive is preceded by a letter, not a boundary, and must not be treated as a quote"
        );
    }

    #[test]
    fn extract_entities_finds_a_single_quoted_span_at_the_very_start_of_the_text() {
        let entities = extract_entities("'opening quote' starts the sentence.");
        assert!(entities.contains(&(EntityType::Quoted, "opening quote".to_string())));
    }

    #[test]
    fn extract_entities_finds_a_single_quoted_span_at_the_very_end_of_the_text() {
        let entities = extract_entities("the sentence ends with 'a closing quote'");
        assert!(entities.contains(&(EntityType::Quoted, "a closing quote".to_string())));
    }

    #[test]
    fn extract_entities_rejects_a_too_short_single_quoted_fragment() {
        let entities = extract_entities("He said 'ok' and left.");
        assert!(
            !entities.iter().any(|(t, _)| *t == EntityType::Quoted),
            "a single-quoted fragment of 2 chars or fewer must be rejected, matching the double-quote length filter"
        );
    }

    #[test]
    fn extract_entities_finds_both_double_and_single_quoted_spans_in_the_same_text() {
        let entities = extract_entities(r#"He said "double quoted" and also 'single quoted' text."#);
        assert!(entities.contains(&(EntityType::Quoted, "double quoted".to_string())));
        assert!(entities.contains(&(EntityType::Quoted, "single quoted".to_string())));
    }

    #[test]
    fn extract_entities_handles_non_ascii_single_quoted_text_without_panicking() {
        let entities = extract_entities("she loves the phrase 'café résumé' a lot.");
        assert!(entities.contains(&(EntityType::Quoted, "café résumé".to_string())));
    }
}
