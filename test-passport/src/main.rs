use bls12_381::Bls12;
use bls12_381::Scalar as Fr;
extern crate rand;

use bellman::{Circuit, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use rand::thread_rng;

use bellman::groth16::{
    create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
};

// Define a constant for the home country's code
const HOME_COUNTRY_CODE: u64 = 100;

// Define a struct representing the Passport Expiration Circuit
#[derive(Clone)]
struct PassportHomeCheckCircuit<F: PrimeField> {
    current_code: Option<F>,
}

impl<F: PrimeField> Circuit<F> for PassportHomeCheckCircuit<F> {
    fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        // Extract the current code value from the struct
        let current_code_value = self.current_code;

        // Allocate a variable for the current code
        let current_code = cs.alloc(
            || "current_code",
            || current_code_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        // Create a constant for the home country's code
        let home_code_value = F::from(HOME_COUNTRY_CODE);
        let home_code = cs.alloc(|| "home_code", || Ok(home_code_value))?;

        // Enforce that the current code is equal to the home country's code
        cs.enforce(
            || "current_code == HOME_COUNTRY_CODE",
            |lc| lc + current_code,
            |lc| lc + CS::one(),
            |lc| lc + home_code,
        );

        Ok(())
    }
}

fn main() {
    // Define the secret value for the current code
    let current_code = Some(Fr::from(100u64));

    // Create an instance of the Passport Expiration Circuit
    let circuit = PassportHomeCheckCircuit { current_code };

    // Generate random parameters for the zk-SNARK proof
    let params = {
        let rng = &mut thread_rng();
        generate_random_parameters::<Bls12, _, _>(circuit.clone(), rng).unwrap()
    };

    // Create a random proof for the circuit
    let proof = {
        let rng = &mut thread_rng();
        create_random_proof(circuit.clone(), &params, rng).unwrap()
    };

    // Prepare the verifying key
    let pvk = prepare_verifying_key(&params.vk);

    // Verify the proof with no public inputs
    let valid = verify_proof(&pvk, &proof, &[]);

    // Check the validity of the proof and print the result
    if valid.is_ok() {
        println!("Current code is the same as the home country's code!");
    } else {
        println!("Current code is not the same as the home country's code!");
    }
}
