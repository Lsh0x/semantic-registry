use crate::traits::Formatter;
use crate::types::Entry;

/// Embarque `text` comme chunk canonique, puis chaque exemple tel quel
pub struct DefaultFormatter;

impl<P> Formatter<P> for DefaultFormatter {
    fn canonical_text(&self, entry: &Entry<P>) -> String {
        entry.text.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_text_then_one_chunk_per_example() {
        let entry =
            Entry::new("id", "canonical", ()).with_examples(vec!["first".into(), "second".into()]);
        assert_eq!(
            DefaultFormatter.format(&entry),
            ["canonical", "first", "second"]
        );
    }
}
