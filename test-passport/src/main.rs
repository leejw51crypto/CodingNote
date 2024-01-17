use bls12_381::Bls12;
use bls12_381::Scalar as Fr;
extern crate rand;

use bellman::{Circuit, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use rand::thread_rng;

use bellman::groth16::{
    create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
};

// Constant for the predefined home country code
const HOME_COUNTRY_CODE: u64 = 100;

// Struct representing the Passport Home Check Circuit
#[derive(Clone)]
struct PassportHomeCheckCircuit<F: PrimeField> {
    // Optional field for the current country code
    current_code: Option<F>,
}

impl<F: PrimeField> Circuit<F> for PassportHomeCheckCircuit<F> {
    fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        // Extract and allocate the current code variable
        let current_code_value = self.current_code;
        let current_code = cs.alloc(
            || "current_code",
            || current_code_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        // Allocate a constant for the home country's code
        let home_code_value = F::from(HOME_COUNTRY_CODE);
        let home_code = cs.alloc(|| "home_code", || Ok(home_code_value))?;

        // Constraint to enforce current code equals home country code
        cs.enforce(
            || "check current_code equals home_code",
            |lc| lc + current_code,
            |lc| lc + CS::one(),
            |lc| lc + home_code,
        );

        Ok(())
    }
}

fn main() {
    // Secret value of the current code for the zk-SNARK proof
    let current_code = Some(Fr::from(100u64));

    // Instantiate the circuit with the current code
    let circuit = PassportHomeCheckCircuit { current_code };

    // Generate parameters for the zk-SNARK proof
    let params = {
        let rng = &mut thread_rng();
        generate_random_parameters::<Bls12, _, _>(circuit.clone(), rng).unwrap()
    };

    // Generate a zk-SNARK proof for the circuit
    let proof = {
        let rng = &mut thread_rng();
        create_random_proof(circuit, &params, rng).unwrap()
    };

    // Prepare the verifying key for the proof
    let pvk = prepare_verifying_key(&params.vk);

    // Verify the zk-SNARK proof with no public inputs
    let valid = verify_proof(&pvk, &proof, &[]);

    // Output the result of the proof verification
    match valid {
        Ok(_) => println!("Proof verified: Current code matches home country code!"),
        Err(_) => println!("Proof failed: Current code does not match home country code!"),
    }
}
