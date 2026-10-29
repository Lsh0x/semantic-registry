//! Intégration MCP (feature `mcp`) : ingestion d'une réponse `tools/list` et définition
//! de l'outil `search_tools`. Le protocole lui-même (transport, serveur) reste à l'appelant.

use serde_json::{json, Value};

use crate::types::Entry;

/// Entrées tirées d'une réponse `tools/list`.
///
/// `id` est le nom de l'outil, `text` son nom, sa description et ses paramètres
/// (voir [`tool_text`]), et `payload` l'objet outil MCP complet, `inputSchema` compris.
/// Les outils sans `name` ou sans `description` sont ignorés.
pub fn parse_tools_list(response: &Value) -> Vec<Entry<Value>> {
    let Some(tools) = response.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    tools
        .iter()
        .filter_map(|tool| {
            let name = tool.get("name")?.as_str()?;
            let description = tool.get("description")?.as_str()?;
            let schema = tool.get("inputSchema").unwrap_or(&Value::Null);
            Some(Entry::new(
                name,
                tool_text(name, description, schema),
                tool.clone(),
            ))
        })
        .collect()
}

/// Texte canonique d'un outil : `"name: description | Parameters: a (doc), b"`
pub fn tool_text(name: &str, description: &str, input_schema: &Value) -> String {
    let mut text = format!("{name}: {description}");
    if let Some(props) = input_schema.get("properties").and_then(Value::as_object) {
        let params: Vec<String> = props
            .iter()
            .map(
                |(k, v)| match v.get("description").and_then(Value::as_str) {
                    Some(d) if !d.is_empty() => format!("{k} ({d})"),
                    _ => k.clone(),
                },
            )
            .collect();
        if !params.is_empty() {
            text.push_str(&format!(" | Parameters: {}", params.join(", ")));
        }
    }
    text
}

/// Définition MCP de l'outil `search_tools`, à déclarer dans un serveur MCP ou une boucle
/// d'agent à la place de la liste complète des outils
pub fn search_tools_definition() -> Value {
    json!({
        "name": "search_tools",
        "description": "Search for available tools dynamically based on user intent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing what tool or action is needed"
                }
            },
            "required": ["query"]
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tools_list_and_skips_incomplete_entries() {
        let response = json!({
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": { "path": { "type": "string", "description": "File path" } }
                    }
                },
                { "name": "no_description" },
                { "name": "no_schema", "description": "Has no inputSchema" }
            ]
        });

        let entries = parse_tools_list(&response);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "read_file");
        assert_eq!(
            entries[0].text,
            "read_file: Read a file | Parameters: path (File path)"
        );
        assert_eq!(
            entries[0].payload["inputSchema"]["properties"]["path"]["type"],
            "string"
        );
        assert_eq!(entries[1].text, "no_schema: Has no inputSchema");
    }

    #[test]
    fn missing_tools_key_yields_nothing() {
        assert!(parse_tools_list(&json!({})).is_empty());
    }

    #[test]
    fn tool_text_lists_parameters_with_and_without_docs() {
        let schema = json!({
            "properties": {
                "city": { "type": "string", "description": "City name" },
                "units": { "type": "string" }
            }
        });
        assert_eq!(
            tool_text("get_weather", "Fetch weather", &schema),
            "get_weather: Fetch weather | Parameters: city (City name), units"
        );
    }

    #[test]
    fn search_tools_requires_a_query() {
        let def = search_tools_definition();
        assert_eq!(def["name"], "search_tools");
        assert_eq!(def["inputSchema"]["required"], json!(["query"]));
    }
}
