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
    let make_moons = make_moons(&mut rng);
    let (training_set, test_set) = make_moons.split_at(100);

    println!("Training set gradient descent");
    let number_of_iterations = 100;
    for k in 0..number_of_iterations {
        let mut loss = Tensor::new(0.0);
        let mut scores = vec![];

        //forward pass
        for (x, y, label) in training_set {
            let score = &model.process(&[*x, *y])[0];
            loss = loss.add(&Tensor::new(*label).sub(score).pow(2.0));
            scores.push((*x, *y, *score.data.borrow()));
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

        let accuracy = accuracy(training_set, &scores);
        println!("Step {}, loss: {}, accuracy: {}%", k, *loss.data.borrow(), accuracy);
    }

    println!("Test set inference");
    let mut scores = vec![];
    for (x, y, _) in test_set {
        let score = &model.process(&[*x, *y])[0];
        scores.push((*x, *y, *score.data.borrow()));
    }

    let accuracy = accuracy(test_set, &scores);
    println!("Accuracy: {}%", accuracy);
}

fn accuracy(labels: &[(f64, f64, f64)], scores: &[(f64, f64, f64)]) -> f64 {
    let matches = labels.iter().zip(scores).filter(|((_, _, label), (_, _, score))|
        (*label > 0.0) == (*score > 0.0)
    ).count();
    matches as f64 / labels.len() as f64 * 100.0
}

fn make_moons(rng: &mut StdRng) -> Vec<(f64, f64, f64)> {
    let mut make_moons = BufReader::new(File::open("make_moons.csv").unwrap()).lines().map(|line| {
        let line = line.unwrap();
        let line: Vec<_> = line.split(',').collect();
        let x = line[0].trim().parse().unwrap();
        let y = line[1].trim().parse().unwrap();
        let label = line[2].trim().parse().unwrap();
        (x, y, label)
    }).collect::<Vec<_>>();
    make_moons.shuffle(rng);
    make_moons
}
