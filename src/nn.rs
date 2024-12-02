use std::rc::Rc;
use crate::autograd::{add, mul, tanh, value, Value};
use rand::{Rng, SeedableRng};
use rand::distributions::Uniform;
use rand::rngs::StdRng;

struct Neuron {
    weights: Vec<Rc<Value>>,
    bias: Rc<Value>,
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Neuron {
    fn process(&self, inputs: &[&Rc<Value>]) -> Rc<Value> {
        let mut sum = value(0.0);
        for (wi, xi) in self.weights.iter().zip(inputs) {
            sum = add(&sum, &mul(wi, xi));
        }
        tanh(&add(&sum, &self.bias))
    }
}

impl Layer {
    fn process(&self, inputs: &[&Rc<Value>]) -> Vec<Rc<Value>> {
        self.neurons.iter().map(|n| n.process(inputs)).collect()
    }
}

fn neuron(number_of_inputs: i32, rng: &mut StdRng) -> Neuron {
    let weights = (0..number_of_inputs).map(|_| value(rng.sample(rng_range()))).collect();
    let bias = value(rng.sample(rng_range()));
    Neuron { weights, bias }
}

fn layer(number_of_neurons: i16, number_of_inputs: i32, rng: &mut StdRng) -> Layer {
    Layer { neurons: (0..number_of_neurons).map(|_| neuron(number_of_inputs, rng)).collect()}
}

fn rng_range() -> Uniform<f64> {
    Uniform::from(-1.0..1.0)
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use crate::autograd::{value};
    use crate::nn::*;

    #[test]
    fn neuron_should_process_input() {
        let mut rng = StdRng::seed_from_u64(42);
        let number_of_inputs = 3;
        let n = neuron(number_of_inputs, &mut rng);
        let inputs: Vec<_> = (0..number_of_inputs).map(|_| value(rng.gen())).collect();
        let inputs_refs: Vec<_> = inputs.iter().collect();

        let output = n.process(&inputs_refs);

        assert_eq!(*output.data.borrow(), 0.050308753080100216);
    }

    #[test]
    fn layer_should_process_input_forwarding_to_all_neurons() {
        let mut rng = StdRng::seed_from_u64(42);
        let number_of_inputs = 3;
        let l = layer(3, number_of_inputs, &mut rng);
        let inputs: Vec<_> = (0..number_of_inputs).map(|_| value(rng.gen())).collect();
        let inputs_refs: Vec<_> = inputs.iter().collect();

        let output = l.process(&inputs_refs);

        let outputs: Vec<_> = output.iter().map(|n| *n.data.borrow()).collect();
        assert_eq!(outputs, vec![
            -0.012655113950167263,
            0.5073009752057568,
            0.03629545163121696,
        ]);
    }
}
