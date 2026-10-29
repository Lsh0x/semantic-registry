use crate::traits::ToolFormatter;
use crate::types::ToolDefinition;

pub struct DefaultToolFormatter;

impl ToolFormatter for DefaultToolFormatter {
    fn format(&self, tool: &ToolDefinition) -> Vec<String> {
        let mut chunks = Vec::new();

        // 1. Chunk principal : Nom + Description
        let mut base_desc = format!("{}: {}", tool.name, tool.description);

        if let Some(props) = tool
            .parameters
            .get("properties")
            .and_then(|p| p.as_object())
        {
            let params: Vec<String> = props
                .iter()
                .map(|(k, v)| {
                    let d = v.get("description").and_then(|d| d.as_str()).unwrap_or("");
                    if d.is_empty() {
                        k.clone()
                    } else {
                        format!("{k} ({d})")
                    }
                })
                .collect();
            if !params.is_empty() {
                base_desc.push_str(&format!(" | Parameters: {}", params.join(", ")));
            }
        }
        chunks.push(base_desc);

        // 2. Un chunk dédié par exemple (sans boilerplate inutile)
        for example in &tool.examples {
            chunks.push(example.clone());
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn base_chunk_lists_parameters_then_one_chunk_per_example() {
        let tool = ToolDefinition::new(
            "get_weather",
            "Fetch weather",
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name" },
                    "units": { "type": "string" }
                }
            }),
        )
        .with_examples(vec!["weather in Paris?".into()]);

        let chunks = DefaultToolFormatter.format(&tool);
        assert_eq!(
            chunks,
            [
                "get_weather: Fetch weather | Parameters: city (City name), units",
                "weather in Paris?",
            ]
        );
    }

    #[test]
    fn tool_without_parameters_has_single_chunk() {
        let tool = ToolDefinition::new("ping", "Health check", json!({}));
        assert_eq!(DefaultToolFormatter.format(&tool), ["ping: Health check"]);
    }
}
