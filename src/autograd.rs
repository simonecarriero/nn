use std::cell::RefCell;
use std::collections::HashSet;
use std::f64::consts::E;
use std::ops;
use std::rc::Rc;
use uuid::Uuid;

pub struct Tensor(Rc<Value>);

#[derive(Debug, PartialEq)]
pub struct Value {
    pub data: RefCell<f64>,
    label: Uuid,
    grad: RefCell<f64>,
    back: Op,
}

#[derive(Debug, PartialEq)]
enum Op {
    None,
    Sum { x: Rc<Value>, y: Rc<Value> },
    Mul { x: Rc<Value>, y: Rc<Value> },
    Pow { x: Rc<Value>, y: Rc<Value> },
    Tanh { x: Rc<Value> },
}

impl Tensor {
    pub fn new(data: f64) -> Tensor {
        Tensor::new_with_grad(data, 0.0)
    }

    pub fn new_with_grad(data: f64, grad: f64) -> Tensor {
        Tensor(value(data, grad))
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor(Rc::new(Value {
            label: Uuid::new_v4(),
            data: RefCell::new(*self.data.borrow() + *other.data.borrow()),
            grad: RefCell::new(0.0),
            back: Op::Sum { x: Rc::clone(self), y: Rc::clone(other) },
        }))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        Tensor(Rc::new(Value {
            label: Uuid::new_v4(),
            data: RefCell::new(*self.data.borrow() * *other.data.borrow()),
            grad: RefCell::new(0.0),
            back: Op::Mul { x: Rc::clone(self), y: Rc::clone(other) },
        }))
    }

    pub fn tanh(&self) -> Tensor {
        let d = *self.data.borrow();
        Tensor(Rc::new(Value {
            label: Uuid::new_v4(),
            data: RefCell::new((E.powf(2.0 * d) - 1.0) / (E.powf(2.0 * d) + 1.0)),
            grad: RefCell::new(0.0),
            back: Op::Tanh { x: Rc::clone(self) },
        }))
    }

    pub fn pow(&self, other: f64) -> Tensor {
        Tensor(Rc::new(Value {
            label: Uuid::new_v4(),
            data: RefCell::new((*self.data.borrow()).powf(other)),
            grad: RefCell::new(0.0),
            back: Op::Pow { x: Rc::clone(self), y: value(other, 0.0) }
        }))
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.add(&other.mul(&Tensor::new(-1.0)))
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        self.mul(&other.pow(-1.0))
    }

    pub fn backward(&self) {
        *self.grad.borrow_mut() = 1.0;
        let mut topo = vec![];
        topological_sort(self, &mut topo, &mut HashSet::new());

        for v in topo.iter().rev() {
            backward_step(v);
        }
    }
}

impl ops::Deref for Tensor {
    type Target = Rc<Value>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn value(data: f64, grad: f64) -> Rc<Value> {
    Rc::new(Value {
        label: Uuid::new_v4(),
        data: RefCell::new(data),
        grad: RefCell::new(grad),
        back: Op::None,
    })
}

fn prev(value: &Rc<Value>) -> Vec<&Rc<Value>> {
   match &value.back {
       Op::None => vec![],
       Op::Sum { x, y } => vec![x, y],
       Op::Mul { x, y } => vec![x, y],
       Op::Pow { x, y } => vec![x, y],
       Op::Tanh { x } => vec![x],
   }
}

fn topological_sort<'a>(value: &'a Rc<Value>, topo: &mut Vec<&'a Rc<Value>>, visited: &mut HashSet<Uuid>) {
    if !visited.contains(&value.label) {
        visited.insert(value.label);
        for v in prev(value) {
            topological_sort(v, topo, visited)
        }
        topo.push(value)
    }
}

fn backward_step(value: &Rc<Value>) {
    match &value.back {
        Op::None => {}
        Op::Sum { x, y } => {
            let x_local_derivative = 1.0;
            let y_local_derivative = 1.0;
            *x.grad.borrow_mut() += x_local_derivative * *value.grad.borrow();
            *y.grad.borrow_mut() += y_local_derivative * *value.grad.borrow();
        }
        Op::Mul { x, y } => {
            let x_local_derivative = *y.data.borrow();
            let y_local_derivative = *x.data.borrow();
            *x.grad.borrow_mut() += x_local_derivative * *value.grad.borrow();
            *y.grad.borrow_mut() += y_local_derivative * *value.grad.borrow();
        }
        Op::Pow { x, y } => {
            let x_local_derivative = *y.data.borrow() * x.data.borrow().powf(*y.data.borrow() - 1.0);
            *x.grad.borrow_mut() += x_local_derivative * *value.grad.borrow();
        }
        Op::Tanh { x } => {
            let x_local_derivative = 1.0 -  &value.data.borrow().powi(2);
            *x.grad.borrow_mut() += x_local_derivative * *value.grad.borrow();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_builds_dag() {
        let a = Tensor::new(0.1_f64.sqrt());
        let b = Tensor::new(0.2);
        let c = Tensor::new(0.3);
        let x = &a.pow(2.0).add(&b.mul(&c)).tanh();

        assert_eq!(*x.data.borrow(), 0.15864850429749888);
        if let Op::Tanh { x } = &x.back {
            assert_eq!(*x.data.borrow(), 0.16);
            if let Op::Sum { x, y } = &x.back {
                assert_eq!(*x.data.borrow(), 0.1);
                if let Op::Pow { x, y } = &x.back {
                    assert_eq!(*x.data.borrow(), 0.1_f64.sqrt());
                    assert_eq!(*y.data.borrow(), 2.0);
                } else {
                    assert!(false, "unexpected Op")
                }
                assert_eq!(*y.data.borrow(), 0.06);
                if let Op::Mul { x, y} = &y.back {
                    assert_eq!(*x.data.borrow(), 0.2);
                    assert_eq!(*y.data.borrow(), 0.3);
                } else {
                    assert!(false, "unexpected Op")
                }
            } else {
                assert!(false, "unexpected Op")
            }
        } else {
            assert!(false, "unexpected Op")
        }
    }

    #[test]
    fn backward_on_sum_should_add_gradient_to_the_variables() {
        let a = Tensor::new_with_grad(0.0, 0.1);
        let b = Tensor::new_with_grad(0.0, 0.2);
        let x = a.add(&b);

        x.backward();

        assert_eq!(*a.grad.borrow(), 1.1);
        assert_eq!(*b.grad.borrow(), 1.2);
    }

    #[test]
    fn backward_on_mul_should_add_gradient_to_the_variables() {
        let a = Tensor::new_with_grad(0.5, 0.05);
        let b = Tensor::new_with_grad(0.25, 0.05);
        let x = a.mul(&b);

        x.backward();

        assert_eq!(*a.grad.borrow(), 0.3);
        assert_eq!(*b.grad.borrow(), 0.55);
    }

    #[test]
    fn backward_on_tanh_should_add_gradient_to_the_variable() {
        let a = Tensor::new_with_grad(0.8814, 0.5);
        let x = a.tanh();

        x.backward();

        assert_eq!(*a.grad.borrow(), 0.9999813233768232);
    }

    #[test]
    fn backward_on_pow_should_add_gradient_to_the_variables() {
        let a = Tensor::new_with_grad(2.0, 0.1);
        let x = a.pow(3.0);

        x.backward();

        assert_eq!(*a.grad.borrow(), 12.1);
    }

    #[test]
    fn backward_on_sub_should_add_gradient_to_the_variables() {
        let a = Tensor::new_with_grad(3.0, 0.5);
        let b = Tensor::new_with_grad(2.0, -0.5);
        let x = a.sub(&b);

        x.backward();

        assert_eq!(*a.grad.borrow(), 1.5);
        assert_eq!(*b.grad.borrow(), -1.5);
    }

    #[test]
    fn backward_on_div_should_add_gradient_to_the_variables() {
        let a = Tensor::new_with_grad(3.0, 0.1);
        let b = Tensor::new_with_grad(2.0, 0.1);
        let x = a.div(&b);

        x.backward();

        assert_eq!(*a.grad.borrow(), 0.6);
        assert_eq!(*b.grad.borrow(), -0.65);
    }

    #[test]
    fn backward_on_dag_should_execute_on_topological_sort() {
        let a = Tensor::new(3.0);
        let b = Tensor::new(3.0);
        let c = Tensor::new(2.0);
        let x = a.mul(&b).add(&c.mul(&a.mul(&b)));

        x.backward();

        assert_eq!(*a.grad.borrow(), 9.0);
        assert_eq!(*b.grad.borrow(), 9.0);
        assert_eq!(*c.grad.borrow(), 9.0);
    }
}
