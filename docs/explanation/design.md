# API design reasoning

## document goal

this document aims to broadly explain the reasoning behind the Pipeline adapter design and how it's implemented.

### `Pipeline` class: abstract goal

abstractly, the `Pipeline` class aims to model and help with implementing pipelines for re-classification of cells in multi condition experiments.
as described in the original HiDDEN publication, this process can be broken down broadly into the following steps:

1) a normalization / variance stabilizing transform step
2) a dimensionality reduction step, which uses the outputs of the previous normalization step
3) a regression step, which given the dimensionality reduction matrix, and an original set of boolean labels, provides a continuous per-cell value indicating the degree to which the cell was affected by the condition
4) a binarization step, which given said per-cell continuous values, binarizes these values back into a set of "adjusted" boolean labels

furthermore, as described by the original HiDDEN manuscript, choosing the desired dimensionality for step 2 is not trivial, and thus we extend this pipeline with a fifth step:

5) a scoring step, where the outputs of the pipeline are used to assess the "quality" of the relabeling, s.t. a set of dimensions can be evaluated for step 2) iteratively and an optimal number can be selected in an automated but data-driven fashion

## `Pipeline` design: main problem

given the above problem statement, it is evident that a concrete instance of a complete analysis can be modeled as the chaining of the individual 5 steps.
however, parametrizing these 5 steps is immediately not evident, as it is our interest that any pipeline can be arbitrarily extended with a new component as new methods of dimensionality reduction, count transformation, etc. are developped.
these new components could have unpredictable inputs, and could further be informed by non RNA-seq methods, such as epigenetic or spatial information, or even currently unknown future modalities of data collection.

as such, it is of interest that any provided step is able to itself "declare" its own data dependencies and have said dependencies be passed to it during pipeline execution.
such ideas are not new in software development, and design of the the `Pipeline` class borrows heavily from "Inversion of Control" design principles.

## `Pipeline` design: concrete implementation

to achieve this inversion of control, the `Pipeline` class specifically inspects the argument names of each provided step function, and provides it data accordingly.

furthermore, outputs from each step are registered a specific "name", that further steps can access if they have a matching argument name.

moreover, when initially calling a pipeline, named arguments are used to set the initial "pipeline variables" which can be accessed by any step in the same fashion as above.
