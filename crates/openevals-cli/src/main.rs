mod run;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use openevals_core::config::{RunConfig, TASK_TYPES};
use openevals_core::registry::Registry;
use openevals_core::report;
use openevals_core::results::ResultsFile;
use openevals_tasks::data::dataset_specs;
use openevals_tasks::DataSource;

#[derive(Parser)]
#[command(
    name = "openevals",
    version,
    about = "Benchmarking suite for open-weight language models"
)]
struct Cli {
    /// Log level: error, warn, info, debug, trace
    #[arg(long, global = true, default_value = "info")]
    log_level: String,
    /// Model registry file. Defaults to the registry compiled into the binary.
    #[arg(long, global = true)]
    registry: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run benchmarks against OpenAI-compatible endpoints
    Run {
        #[arg(short, long)]
        config: PathBuf,
        /// Only run these model entries (labels from the config)
        #[arg(long, num_args = 1..)]
        models: Vec<String>,
        /// Only run these tasks (names from the config)
        #[arg(long, num_args = 1..)]
        tasks: Vec<String>,
        /// Override run.output
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Override the item limit of every task
        #[arg(long)]
        limit: Option<usize>,
        /// Print the plan and exit
        #[arg(long)]
        dry_run: bool,
    },
    /// List models in the registry
    Models {
        /// Filter by family substring
        #[arg(long)]
        family: Option<String>,
        #[arg(long, value_enum, default_value = "table")]
        format: ModelFormat,
    },
    /// List available benchmark task types
    Tasks,
    /// Download and cache benchmark datasets
    Download {
        /// Task types to download (default: all)
        #[arg(num_args = 0..)]
        tasks: Vec<String>,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
    },
    /// Render a leaderboard from a results file or output directory
    Report {
        path: PathBuf,
        #[arg(long, value_enum, default_value = "md")]
        format: ReportFormat,
    },
    /// Validate a run configuration
    Validate {
        #[arg(short, long)]
        config: PathBuf,
    },
}

#[derive(Clone, ValueEnum)]
enum ModelFormat {
    Table,
    Json,
}

#[derive(Clone, ValueEnum)]
enum ReportFormat {
    Md,
    Csv,
    Json,
}

fn load_registry(path: &Option<PathBuf>) -> Result<Registry> {
    Ok(match path {
        Some(p) => {
            Registry::from_path(p).with_context(|| format!("loading registry {}", p.display()))?
        }
        None => Registry::embedded(),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&cli.log_level)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let registry = load_registry(&cli.registry)?;

    match cli.command {
        Command::Run {
            config,
            models,
            tasks,
            output,
            limit,
            dry_run,
        } => {
            let cfg = RunConfig::from_path(&config)?;
            run::execute(
                cfg,
                &registry,
                run::Filters {
                    models,
                    tasks,
                    output,
                    limit,
                    dry_run,
                },
            )
            .await
        }
        Command::Models { family, format } => {
            let list: Vec<_> = match &family {
                Some(f) => registry.by_family(f),
                None => registry.all().iter().collect(),
            };
            match format {
                ModelFormat::Json => println!("{}", serde_json::to_string_pretty(&list)?),
                ModelFormat::Table => {
                    println!(
                        "{:<28} {:<14} {:>9} {:>9} {:>9}  {:<8} HF repo",
                        "id", "family", "params_b", "active_b", "context", "thinking"
                    );
                    for m in &list {
                        println!(
                            "{:<28} {:<14} {:>9} {:>9} {:>9}  {:<8} {}",
                            m.id,
                            m.family,
                            m.params_b,
                            m.active_params_b,
                            m.context_length,
                            format!("{:?}", m.reasoning).to_lowercase(),
                            m.hf_repo
                        );
                    }
                    println!("{} models", list.len());
                }
            }
            Ok(())
        }
        Command::Tasks => {
            for t in TASK_TYPES {
                println!("{t}");
            }
            Ok(())
        }
        Command::Download { tasks, data_dir } => {
            let ds = DataSource::new(&data_dir);
            let wanted: Vec<String> = if tasks.is_empty() {
                TASK_TYPES.iter().map(|s| s.to_string()).collect()
            } else {
                tasks
            };
            for t in &wanted {
                if !TASK_TYPES.contains(&t.as_str()) {
                    bail!("unknown task type '{t}'");
                }
                for (dataset, config, split) in dataset_specs(t) {
                    match ds.rows(dataset, config, split).await {
                        Ok(rows) => println!("{dataset} {config} {split}: {} rows", rows.len()),
                        Err(e) => eprintln!("{dataset} {config} {split}: {e}"),
                    }
                }
            }
            Ok(())
        }
        Command::Report { path, format } => {
            let file = if path.is_dir() {
                path.join("results.json")
            } else {
                path
            };
            let results =
                ResultsFile::load(&file).with_context(|| format!("reading {}", file.display()))?;
            match format {
                ReportFormat::Md => print!("{}", report::markdown_leaderboard(&results)),
                ReportFormat::Csv => print!("{}", report::csv(&results)),
                ReportFormat::Json => println!("{}", serde_json::to_string_pretty(&results)?),
            }
            Ok(())
        }
        Command::Validate { config } => {
            let cfg = RunConfig::from_path(&config)?;
            for m in &cfg.models {
                if registry.get(&m.model).is_none() && m.served_name.is_none() {
                    bail!(
                        "model '{}' is not in the registry and has no served_name",
                        m.model
                    );
                }
            }
            println!("ok: {} models, {} tasks", cfg.models.len(), cfg.tasks.len());
            Ok(())
        }
    }
}
