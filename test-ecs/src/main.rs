use bevy::prelude::*;
use chrono::{DateTime, Utc};
use fake::faker::company::en::{CompanyName, Industry};
use fake::faker::currency::en::CurrencyCode;
use fake::faker::name::en::Name;
use fake::{Fake, Faker};
use rand::seq::SliceRandom;
use rand::Rng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// Components
#[derive(Component, Debug, Clone)]
struct Transaction {
    from: String,
    to: String,
    amount: f64,
    currency: String,
    timestamp: DateTime<Utc>,
    transaction_type: TransactionType,
    is_pending: bool,
}

#[derive(Debug, Clone)]
enum TransactionType {
    Payment,
    Investment,
    Transfer,
    SmartContract,
}

#[derive(Component, Debug)]
struct Block {
    index: u64,
    timestamp: DateTime<Utc>,
    previous_hash: String,
    hash: String,
    nonce: u64,
    miner: String,
    mining_reward: f64,
}

#[derive(Component)]
struct PendingBlock;

#[derive(Component)]
struct MinedBlock;

#[derive(Component)]
struct BlockchainState {
    blocks_to_generate: u32,
    participants: Vec<Participant>,
    current_index: u64,
    network_hashrate: f64,
    difficulty: u32,
}

#[derive(Clone, Debug)]
struct Participant {
    name: String,
    company: String,
    industry: String,
    balance: f64,
}

// Resource for timing
#[derive(Resource)]
struct BlockTimer {
    timer: Timer,
}

#[derive(Resource)]
struct BlockchainResource {
    has_pending_block: bool,
    has_pending_transactions: bool,
}

impl Block {
    fn calculate_hash(&self, transactions: &[Entity]) -> String {
        let mut hasher = Sha256::new();
        let input = format!(
            "{}{:?}{:?}{}{}",
            self.index, self.timestamp, transactions, self.previous_hash, self.nonce
        );
        hasher.update(input.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn mine_block(&mut self, transactions: &[Entity], difficulty: u32) {
        let target = "0".repeat(difficulty as usize);
        while &self.hash[0..difficulty as usize] != target.as_str() {
            self.nonce += 1;
            self.hash = self.calculate_hash(transactions);
        }
        println!("Block mined! Nonce: {}, Hash: {}", self.nonce, self.hash);
    }
}

fn generate_random_participant() -> Participant {
    Participant {
        name: Name().fake(),
        company: CompanyName().fake(),
        industry: Industry().fake(),
        balance: (1000.0..100000.0).fake(),
    }
}

fn generate_random_transactions(participants: &[Participant], count: usize) -> Vec<Transaction> {
    let mut rng = rand::thread_rng();
    let mut transactions = Vec::new();
    let transaction_types = vec![
        TransactionType::Payment,
        TransactionType::Investment,
        TransactionType::Transfer,
        TransactionType::SmartContract,
    ];

    for _ in 0..count {
        let from = participants.choose(&mut rng).unwrap();
        let mut to = participants.choose(&mut rng).unwrap();

        while to.name == from.name {
            to = participants.choose(&mut rng).unwrap();
        }

        let amount = (10.0..5000.0).fake();
        let transaction_type = transaction_types.choose(&mut rng).unwrap().clone();

        transactions.push(Transaction {
            from: from.name.clone(),
            to: to.name.clone(),
            amount,
            currency: CurrencyCode().fake(),
            timestamp: Utc::now(),
            transaction_type,
            is_pending: true,
        });
    }

    transactions
}

// Systems
fn setup(mut commands: Commands) {
    // Generate random participants
    let participants: Vec<Participant> = (0..10).map(|_| generate_random_participant()).collect();

    // Create blockchain state
    let state = BlockchainState {
        blocks_to_generate: 10,
        participants,
        current_index: 0,
        network_hashrate: 1000.0,
        difficulty: 2,
    };
    commands.spawn(state);

    // Create genesis block
    let genesis_block = Block {
        index: 0,
        timestamp: Utc::now(),
        previous_hash: "0".to_string(),
        hash: "0".to_string(),
        nonce: 0,
        miner: "Genesis".to_string(),
        mining_reward: 0.0,
    };
    commands.spawn((genesis_block, MinedBlock));

    // Add resources
    commands.insert_resource(BlockTimer {
        timer: Timer::from_seconds(1.0, TimerMode::Repeating),
    });
    commands.insert_resource(BlockchainResource {
        has_pending_block: false,
        has_pending_transactions: false,
    });
}

fn generate_transactions(
    mut commands: Commands,
    state_query: Query<&BlockchainState>,
    mut blockchain_resource: ResMut<BlockchainResource>,
    pending_transactions: Query<&Transaction, With<Transaction>>,
) {
    if !blockchain_resource.has_pending_transactions && !blockchain_resource.has_pending_block {
        if let Ok(state) = state_query.get_single() {
            if state.blocks_to_generate > 0 {
                let mut rng = rand::thread_rng();
                let transaction_count = rng.gen_range(1..=5);
                let transactions =
                    generate_random_transactions(&state.participants, transaction_count);

                println!(
                    "\nGenerating block #{} with {} transactions",
                    state.current_index + 1,
                    transaction_count
                );

                // Spawn transaction entities
                for transaction in transactions {
                    commands.spawn(transaction);
                }
                blockchain_resource.has_pending_transactions = true;
            }
        }
    }
}

fn create_pending_block(
    mut commands: Commands,
    state_query: Query<&BlockchainState>,
    transaction_query: Query<(Entity, &Transaction), With<Transaction>>,
    block_query: Query<&Block, With<MinedBlock>>,
    mut blockchain_resource: ResMut<BlockchainResource>,
) {
    if blockchain_resource.has_pending_transactions && !blockchain_resource.has_pending_block {
        if let Ok(state) = state_query.get_single() {
            if state.blocks_to_generate > 0 {
                let pending_transactions: Vec<_> = transaction_query
                    .iter()
                    .filter(|(_, tx)| tx.is_pending)
                    .map(|(e, _)| e)
                    .collect();

                if !pending_transactions.is_empty() {
                    let mut rng = rand::thread_rng();
                    let miner = state.participants.choose(&mut rng).unwrap();
                    let previous_block = block_query.iter().last().unwrap();

                    let new_block = Block {
                        index: state.current_index + 1,
                        timestamp: Utc::now(),
                        previous_hash: previous_block.hash.clone(),
                        hash: String::new(),
                        nonce: 0,
                        miner: miner.name.clone(),
                        mining_reward: 10.0 / (2.0f64.powf((state.current_index / 10) as f64)),
                    };

                    commands.spawn((new_block, PendingBlock));
                    blockchain_resource.has_pending_block = true;
                }
            }
        }
    }
}

fn mine_block(
    mut commands: Commands,
    mut state_query: Query<&mut BlockchainState>,
    mut block_query: Query<(Entity, &mut Block, &PendingBlock)>,
    transaction_query: Query<(Entity, &Transaction), With<Transaction>>,
    time: Res<Time>,
    mut block_timer: ResMut<BlockTimer>,
    mut blockchain_resource: ResMut<BlockchainResource>,
) {
    block_timer.timer.tick(time.delta());

    if block_timer.timer.just_finished() && blockchain_resource.has_pending_block {
        if let Ok(mut state) = state_query.get_single_mut() {
            if let Ok((block_entity, mut block, _)) = block_query.get_single_mut() {
                let pending_transactions: Vec<_> = transaction_query
                    .iter()
                    .filter(|(_, tx)| tx.is_pending)
                    .map(|(e, _)| e)
                    .collect();

                if !pending_transactions.is_empty() {
                    println!(
                        "Mining block with difficulty: {} (Network Hashrate: {:.2} H/s)",
                        state.difficulty, state.network_hashrate
                    );

                    block.hash = block.calculate_hash(&pending_transactions);
                    block.mine_block(&pending_transactions, state.difficulty);

                    println!("Block successfully mined by {}!", block.miner);
                    println!("\nBlock #{}", block.index);
                    println!("Miner: {}", block.miner);
                    println!("Mining Reward: {:.8} BTC", block.mining_reward);
                    println!("Hash: {}", block.hash);
                    println!("Previous Hash: {}", block.previous_hash);
                    println!("Transactions:");

                    for (entity, tx) in transaction_query.iter() {
                        if tx.is_pending {
                            println!(
                                "  {} -> {}: {:.2} {} ({:?})",
                                tx.from, tx.to, tx.amount, tx.currency, tx.transaction_type
                            );
                            // Mark transaction as processed
                            commands.entity(entity).remove::<Transaction>();
                        }
                    }
                    println!("-------------------");

                    // Update state
                    state.current_index += 1;
                    state.blocks_to_generate -= 1;
                    state.network_hashrate *= 1.1;

                    // Convert PendingBlock to MinedBlock
                    commands.entity(block_entity).remove::<PendingBlock>();
                    commands.entity(block_entity).insert(MinedBlock);

                    // Add mining reward transaction
                    commands.spawn(Transaction {
                        from: "System".to_string(),
                        to: block.miner.clone(),
                        amount: block.mining_reward,
                        currency: "BTC".to_string(),
                        timestamp: Utc::now(),
                        transaction_type: TransactionType::SmartContract,
                        is_pending: true,
                    });

                    // Reset blockchain resource state
                    blockchain_resource.has_pending_block = false;
                    blockchain_resource.has_pending_transactions = false;

                    if state.blocks_to_generate == 0 {
                        print_final_statistics(
                            &block_query.iter().map(|(_, b, _)| b).collect::<Vec<_>>(),
                            &transaction_query.iter().map(|(_, t)| t).collect::<Vec<_>>(),
                            &state,
                        );
                    }
                }
            }
        }
    }
}

fn print_final_statistics(
    blocks: &[&Block],
    transactions: &[&Transaction],
    state: &BlockchainState,
) {
    println!("\nBlockchain generation complete!");
    println!("Total blocks: {}", blocks.len());
    println!("Final network hashrate: {:.2} H/s", state.network_hashrate);

    let mut mining_rewards = HashMap::new();
    let mut participant_stats = HashMap::new();

    for block in blocks.iter().skip(1) {
        mining_rewards
            .entry(block.miner.clone())
            .and_modify(|r| *r += block.mining_reward)
            .or_insert(block.mining_reward);
    }

    for tx in transactions {
        if tx.from != "System" {
            participant_stats
                .entry(tx.from.clone())
                .and_modify(|s: &mut (f64, u32)| {
                    s.0 += tx.amount;
                    s.1 += 1;
                })
                .or_insert((tx.amount, 1));
        }
    }

    println!("\nMining Rewards Summary:");
    println!("------------------------");
    let mut rewards: Vec<_> = mining_rewards.iter().collect();
    rewards.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (miner, reward) in rewards {
        println!("{}: {:.8} BTC", miner, reward);
    }

    println!("\nParticipant Statistics:");
    println!("------------------------");
    let mut stats: Vec<_> = participant_stats.iter().collect();
    stats.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    for (participant, (volume, count)) in stats {
        println!("{}: {:.2} ({} transactions)", participant, volume, count);
    }
}

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (generate_transactions, create_pending_block, mine_block).chain(),
        )
        .run();
}
