use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "amm-lab", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a pool and show initial state
    Pool {
        #[arg(long)]
        reserve_x: u128,
        #[arg(long)]
        reserve_y: u128,
        #[arg(long)]
        fee_bps: u16,
    },
    /// Execute a swap and show receipt
    Swap {
        #[arg(long)]
        reserve_x: u128,
        #[arg(long)]
        reserve_y: u128,
        #[arg(long)]
        direction: String, // "x-to-y" or "y-to-x"
        #[arg(long)]
        amount_in: u128,
        #[arg(long)]
        fee_bps: u16,
    },
    Scenario {
        #[command(subcommand)]
        action: ScenarioAction,
    },
}

#[derive(Subcommand)]
enum ScenarioAction {
    Run {
        #[arg()]
        file: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Pool {
            reserve_x,
            reserve_y,
            fee_bps,
        } => match amm_lab::pool::Pool::new(reserve_x, reserve_y, fee_bps) {
            Ok(pool) => {
                println!("Pool created");
                println!("  reserve_x:  {}", pool.reserve_x);
                println!("  reserve_y:  {}", pool.reserve_y);
                println!("  lp_supply:  {}", pool.lp_supply);
                println!("  fee_bps:    {}", pool.fee_bps);
                println!("  spot_price: {:.6}", pool.spot_price());
                println!("  invariant:  {}", pool.invariant());
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        },
        Commands::Swap {
            reserve_x,
            reserve_y,
            direction,
            amount_in,
            fee_bps,
        } => {
            let dir = match direction.as_str() {
                "x-to-y" => amm_lab::swap::SwapDirection::XtoY,
                "y-to-x" => amm_lab::swap::SwapDirection::YtoX,
                _ => {
                    eprintln!("Error: direction must be 'x-to-y' or 'y-to-x'");
                    std::process::exit(1);
                }
            };
            let mut pool = match amm_lab::pool::Pool::new(reserve_x, reserve_y, fee_bps) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            };
            match amm_lab::swap::swap(&mut pool, dir, amount_in) {
                Ok(receipt) => {
                    println!("Swap executed");
                    println!(" direction:         {:?}", receipt.quote.direction);
                    println!(" amount_in:         {}", receipt.quote.amount_in);
                    println!(" fee_amount:        {}", receipt.quote.fee_amount);
                    println!(" amount_out:        {}", receipt.quote.amount_out);
                    println!(" spot_price_before: {:.6}", receipt.quote.spot_price_before);
                    println!(" exec_price:        {:.6}", receipt.quote.exec_price);
                    println!(" price_impact:      {:.4}%", receipt.quote.price_impact_pct);
                    println!(
                        " invariant_before:        {}",
                        receipt.quote.invariant_before
                    );
                    println!(" invariant_after:        {}", receipt.quote.invariant_after);
                    println!(" reserve_x_after:        {}", receipt.reserve_x_after);
                    println!(" reserve_y_after:        {}", receipt.reserve_y_after);
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Scenario { action } => match action {
            ScenarioAction::Run { file } => {
                let path = std::path::Path::new(&file);
                let scenario = amm_lab::scenario::load_scenario(path).unwrap_or_else(|e| {
                    eprintln!("Failed to load scenario: {e}");
                    std::process::exit(1);
                });
                let report = amm_lab::scenario::run_scenario(&scenario).unwrap_or_else(|e| {
                    eprintln!("Scenario error: {e}");
                    std::process::exit(1);
                });

                for line in &report.log {
                    println!("{line}");
                }

                let out_dir = std::path::Path::new("data/processed");
                match amm_lab::scenario::write_json(&report, out_dir) {
                    Ok(p) => println!("\nJSON → {p}"),
                    Err(e) => eprintln!("JSON write error: {e}"),
                }
                match amm_lab::scenario::write_csv_swaps(&report, out_dir) {
                    Ok(p) => println!("CSV  → {p}"),
                    Err(e) => eprintln!("CSV write error: {e}"),
                }
            }
        },
    }
}
