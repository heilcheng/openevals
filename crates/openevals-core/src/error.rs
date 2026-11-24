use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("http error: {0}")]
    Http(String),
    #[error("server returned status {status}: {body}")]
    Status {
        status: u16,
        body: String,
        retry_after_secs: Option<u64>,
    },
    #[error("invalid response: {0}")]
    Parse(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("configuration error: {0}")]
    Config(String),
    #[error("dataset error: {0}")]
    Data(String),
}

impl Error {
    /// Whether a retry could plausibly succeed.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Http(_) => true,
            Error::Status { status, .. } => *status == 429 || *status == 408 || *status >= 500,
            _ => false,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Parse(e.to_string())
    }
}
