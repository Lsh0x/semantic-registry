use crate::types::ToolDefinition;
use serde_json::json;

pub struct MetaTool;

impl MetaTool {
    pub fn spec() -> ToolDefinition {
        ToolDefinition::new(
            "search_tools",
            "Search for available tools dynamically based on user intent.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what tool or action is needed"
                    }
                },
                "required": ["query"]
            }),
        )
    }

    pub fn search_tools_definition() -> ToolDefinition {
        Self::spec()
    }
}
