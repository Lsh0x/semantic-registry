use crate::types::ToolDefinition;
use serde_json::Value;

pub struct McpParser;

impl McpParser {
    /// Extrait les ToolDefinitions depuis la réponse JSON d'un appel `tools/list` de MCP
    pub fn parse_tools_list(json_val: &Value) -> Vec<ToolDefinition> {
        let mut tools = Vec::new();
        if let Some(tools_array) = json_val.get("tools").and_then(|t| t.as_array()) {
            for tool in tools_array {
                if let (Some(name), Some(desc)) = (tool.get("name"), tool.get("description")) {
                    tools.push(ToolDefinition::new(
                        name.as_str().unwrap_or_default(),
                        desc.as_str().unwrap_or_default(),
                        tool.get("inputSchema").cloned().unwrap_or_default(),
                    ));
                }
            }
        }
        tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_mcp_tools_list_and_skips_incomplete_entries() {
        let response = json!({
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "inputSchema": { "type": "object", "properties": { "path": { "type": "string" } } }
                },
                { "name": "no_description" },
                { "name": "no_schema", "description": "Has no inputSchema" }
            ]
        });

        let tools = McpParser::parse_tools_list(&response);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].parameters["properties"]["path"]["type"], "string");
        assert_eq!(tools[1].name, "no_schema");
        assert!(tools[1].parameters.is_null());
    }

    #[test]
    fn missing_tools_key_yields_nothing() {
        assert!(McpParser::parse_tools_list(&json!({})).is_empty());
    }
}
