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

struct Mlp {
    layers: Vec<Layer>
}

impl Neuron {
    fn new(number_of_inputs: i32, rng: &mut StdRng) -> Neuron {
        let weights = (0..number_of_inputs).map(|_| value(rng.sample(rng_range()))).collect();
        let bias = value(rng.sample(rng_range()));
        Neuron { weights, bias }
    }

    fn process(&self, inputs: &[Rc<Value>]) -> Rc<Value> {
        let mut sum = value(0.0);
        for (wi, xi) in self.weights.iter().zip(inputs) {
            sum = add(&sum, &mul(wi, xi));
        }
        tanh(&add(&sum, &self.bias))
    }
}

impl Layer {
    fn new(number_of_inputs: i32, number_of_neurons: i32, rng: &mut StdRng) -> Layer {
        Layer { neurons: (0..number_of_neurons).map(|_| Neuron::new(number_of_inputs, rng)).collect()}
    }

    fn process(&self, inputs: &[Rc<Value>]) -> Vec<Rc<Value>> {
        self.neurons.iter().map(|n| n.process(inputs)).collect()
    }
}

impl Mlp {
    fn new(number_of_inputs: i32, layers: Vec<i32>, rng: &mut StdRng) -> Mlp {
        let n = [vec![number_of_inputs], layers].concat();
        let layers = n.iter().zip(n.iter().skip(1))
            .map(|(n_inputs, n_neurons)| Layer::new(*n_inputs, *n_neurons, rng)).collect();
        Mlp { layers }
    }

    fn process(&self, inputs: &[f64]) -> Vec<Rc<Value>> {
        let mut x: Vec<_> = inputs.iter().map(|i| value(*i)).collect();
        for layer in &self.layers {
            x = layer.process(&x);
        }
        x
    }
}

fn rng_range() -> Uniform<f64> {
    Uniform::from(-1.0..1.0)
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use crate::autograd::value;
    use crate::nn::*;

    #[test]
    fn neuron_should_process_input() {
        let mut rng = StdRng::seed_from_u64(42);
        let number_of_inputs = 3;
        let neuron = Neuron::new(number_of_inputs, &mut rng);
        let inputs: Vec<_> = (0..number_of_inputs).map(|_| value(rng.gen())).collect();

        let output = neuron.process(&inputs);

        assert_eq!(*output.data.borrow(), 0.050308753080100216);
    }

    #[test]
    fn layer_should_process_input_forwarding_to_all_neurons() {
        let mut rng = StdRng::seed_from_u64(42);
        let number_of_inputs = 3;
        let number_of_neurons = 3;
        let layer = Layer::new(number_of_inputs, number_of_neurons, &mut rng);
        let inputs: Vec<_> = (0..number_of_inputs).map(|_| value(rng.gen())).collect();

        let output = layer.process(&inputs);

        let outputs: Vec<_> = output.iter().map(|n| *n.data.borrow()).collect();
        assert_eq!(outputs, vec![
            -0.012655113950167263,
            0.5073009752057568,
            0.03629545163121696,
        ]);
    }

    #[test]
    fn mlp_should_process_inputs() {
        let number_of_inputs = 3;
        let mut rng = StdRng::seed_from_u64(42);
        let mlp = Mlp::new(number_of_inputs, vec![4, 4, 1], &mut rng);

        let inputs: Vec<_> = (0..number_of_inputs).map(|_| rng.gen()).collect();
        let output = mlp.process(&inputs);

        assert_eq!(*output[0].data.borrow(), 0.88226677498484760);
    }
}
