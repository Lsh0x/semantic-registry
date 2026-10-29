use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::tokio::ApiBuilder;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::errors::{Result, ToolRagError};
use crate::traits::Embedder;

#[derive(Clone)]
pub struct CandleEmbedder {
    model: Arc<BertModel>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
}

impl CandleEmbedder {
    /// Initialise le modèle 'all-MiniLM-L6-v2' depuis Hugging Face Hub (avec cache local)
    pub async fn minilm_v2() -> Result<Self> {
        // Contourne le bug reqwest/macOS en ignorant les proxies système mal formés
        if std::env::var("HTTP_PROXY").map_or(true, |v| !v.starts_with("http"))
            && std::env::var("HTTPS_PROXY").map_or(true, |v| !v.starts_with("http"))
        {
            std::env::set_var("NO_PROXY", "*");
        }

        let api = ApiBuilder::new()
            .build()
            .map_err(|e| ToolRagError::TransportError(e.to_string()))?;

        let repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());

        let config_filename = repo
            .get("config.json")
            .await
            .map_err(|e| ToolRagError::TransportError(e.to_string()))?;
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| ToolRagError::TransportError(e.to_string()))?;
        let weights_filename = repo
            .get("model.safetensors")
            .await
            .map_err(|e| ToolRagError::TransportError(e.to_string()))?;

        let config_str = tokio::fs::read_to_string(config_filename)
            .await
            .map_err(|e| ToolRagError::ParseError(e.to_string()))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| ToolRagError::ParseError(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| ToolRagError::ParseError(e.to_string()))?;

        let device = Device::Cpu;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)
                .map_err(|e| ToolRagError::EmbeddingFailed(e.to_string()))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| ToolRagError::EmbeddingFailed(e.to_string()))?;

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }

    /// Extrait l'embedding par Mean Pooling + L2 Normalization
    fn infer(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ToolRagError::EmbeddingFailed(e.to_string()))?;

        let token_ids = tokens.get_ids();
        let token_ids_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;

        let token_type_ids = Tensor::zeros_like(&token_ids_tensor)?;

        let embeddings = self
            .model
            .forward(&token_ids_tensor, &token_type_ids, None)?;

        let (_b_size, seq_len, _hidden_size) = embeddings.dims3()?;

        let sum_embeddings = embeddings.sum(1)?;
        let mean_pooled = (sum_embeddings / (seq_len as f64))?;

        let norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?;

        let normalized = mean_pooled.broadcast_div(&norm)?;

        let vec = normalized.squeeze(0)?.to_vec1::<f32>()?;

        Ok(vec)
    }
}

#[async_trait]
impl Embedder for CandleEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embedder = self.clone();
        let text = text.to_string();

        tokio::task::spawn_blocking(move || embedder.infer(&text))
            .await
            .map_err(|e| ToolRagError::Internal(e.to_string()))?
    }
}
