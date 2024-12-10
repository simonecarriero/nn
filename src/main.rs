use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use crate::autograd::Tensor;
use crate::nn::Mlp;

mod autograd;
mod nn;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let model = Mlp::new(2, vec![16, 16, 1], &mut rng);

    let training_set_size = 100;
    let number_of_iterations = 100;
    let mut make_moons = read_make_moons();
    make_moons.shuffle(&mut rng);

    let training_set = &make_moons[0..training_set_size];
    let test_set = &make_moons[training_set_size..make_moons.len()];

    // Gradient descent
    for k in 0..number_of_iterations {
        let mut loss = Tensor::new(0.0);
        let mut matches = 0;

        //forward pass
        for (x, y, label) in training_set {
            let score = &model.process(&[*x,*y])[0];
            loss = loss.add(&Tensor::new(*label).sub(score).pow(2.0));
            if (*label > 0.0) == (*score.data.borrow() > 0.0) {
                matches += 1;
            }
        }
        loss = loss.mul(&Tensor::new(1.0).div(&Tensor::new(training_set.len() as f64)));

        //backward pass
        model.zero_grad();
        loss.backward();

        //update
        let learning_rate = 1.0 - 0.9 * (k as f64) / number_of_iterations as f64;
        for p in model.parameters() {
            *p.data.borrow_mut() -= *p.grad.borrow() * learning_rate;
        }

        println!("Step {}, loss: {}, accuracy: {}%", k, *loss.data.borrow(), matches as f64 / training_set.len() as f64 * 100.0);
    }

    // Test set
    let mut matches = 0;
    for (x, y, label) in test_set {
        let score = &model.process(&[*x,*y])[0];
        if (*label > 0.0) == (*score.data.borrow() > 0.0) {
            matches += 1;
        }
    }
    println!();
    println!("Test set accuracy: {}%", matches as f64 / test_set.len() as f64 * 100.0);

}

fn read_make_moons() -> Vec<(f64, f64, f64)> {
    BufReader::new(File::open("make_moons.csv").unwrap()).lines().map(|line| {
        let line = line.unwrap();
        let line: Vec<_> = line.split(',').collect();
        let x = line[0].trim().parse().unwrap();
        let y = line[1].trim().parse().unwrap();
        let label = line[2].trim().parse().unwrap();
        (x, y, label)
    }).collect()
}
