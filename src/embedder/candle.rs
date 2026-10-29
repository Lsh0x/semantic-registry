use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::tokio::ApiBuilder;
use std::sync::Arc;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

use crate::error::{Error, Result};
use crate::traits::Embedder;

/// Local sentence embeddings with [Candle](https://github.com/huggingface/candle), on CPU.
///
/// Batches are padded to their longest text and mean pooling ignores the padding, so a
/// text gets the same vector alone or in a batch.
#[derive(Clone)]
pub struct CandleEmbedder {
    model: Arc<BertModel>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
}

impl CandleEmbedder {
    /// Loads sentence-transformers/all-MiniLM-L6-v2 from the Hugging Face Hub, cached
    /// locally after the first download: 384 dimensions, L2-normalized.
    ///
    /// # Errors
    /// The model cannot be downloaded or loaded.
    pub async fn minilm_v2() -> Result<Self> {
        // Works around reqwest on macOS picking up malformed system proxy settings
        if std::env::var("HTTP_PROXY").map_or(true, |v| !v.starts_with("http"))
            && std::env::var("HTTPS_PROXY").map_or(true, |v| !v.starts_with("http"))
        {
            std::env::set_var("NO_PROXY", "*");
        }

        let api = ApiBuilder::new()
            .build()
            .map_err(|e| Error::TransportError(e.to_string()))?;

        let repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());

        let config_filename = repo
            .get("config.json")
            .await
            .map_err(|e| Error::TransportError(e.to_string()))?;
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| Error::TransportError(e.to_string()))?;
        let weights_filename = repo
            .get("model.safetensors")
            .await
            .map_err(|e| Error::TransportError(e.to_string()))?;

        let config_str = tokio::fs::read_to_string(config_filename)
            .await
            .map_err(|e| Error::ParseError(e.to_string()))?;
        let config: Config =
            serde_json::from_str(&config_str).map_err(|e| Error::ParseError(e.to_string()))?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| Error::ParseError(e.to_string()))?;
        // MiniLM's tokenizer.json pads every input to 128 tokens. Pad batches to their
        // longest text instead; the attention mask keeps padding out of the pooling.
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));

        let device = Device::Cpu;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)
                .map_err(|e| Error::EmbeddingFailed(e.to_string()))?
        };

        let model =
            BertModel::load(vb, &config).map_err(|e| Error::EmbeddingFailed(e.to_string()))?;

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }

    /// Mean pooling over real tokens, then L2 normalization, for a whole batch
    fn infer(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::EmbeddingFailed(e.to_string()))?;

        let ids: Vec<Tensor> = encodings
            .iter()
            .map(|e| Tensor::new(e.get_ids(), &self.device))
            .collect::<candle_core::Result<_>>()?;
        let masks: Vec<Tensor> = encodings
            .iter()
            .map(|e| Tensor::new(e.get_attention_mask(), &self.device))
            .collect::<candle_core::Result<_>>()?;
        let input_ids = Tensor::stack(&ids, 0)?;
        let attention_mask = Tensor::stack(&masks, 0)?;
        let token_type_ids = input_ids.zeros_like()?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // (batch, tokens, 1): 1.0 for real tokens, 0.0 for padding
        let mask = attention_mask.to_dtype(hidden.dtype())?.unsqueeze(2)?;
        let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
        let counts = mask.sum(1)?;
        let mean = summed.broadcast_div(&counts)?;

        let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
        Ok(mean.broadcast_div(&norm)?.to_vec2::<f32>()?)
    }
}

#[async_trait]
impl Embedder for CandleEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embed_batch(&[text.to_string()]).await?;
        vectors
            .pop()
            .ok_or_else(|| Error::EmbeddingFailed("empty batch result".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embedder = self.clone();
        let texts = texts.to_vec();

        tokio::task::spawn_blocking(move || embedder.infer(&texts))
            .await
            .map_err(|e| Error::Internal(e.to_string()))?
    }
}
