py <- reticulate::import("found", convert = FALSE)

#' adapters module from python package
#'
#' @export
adapters <- py$adapters

#' methods module from python package
#'
#' @export
methods <- py$methods

#' tune module from python package
#'
#' @export
tune <- py$tune

#' wrapper function which allows injection of R functions into found Pipelines
#'
#' @param fn function
#'
#' @export
to_step <- function(fn) {
  fr <- formals(args(fn))
  if (is.null(fr)) {
    fn_sig <- ""
    fn_pa <- ""
  } else {
    fn_sig <- reticulate:::get_signature(fr)
    # strip any defaults for call expression
    fn_pa <- length(fr) |>
      replicate(quote(expr = ), simplify = FALSE) |>
      stats::setNames(names(fr)) |>
      reticulate:::get_signature()
  }
  py_str <- tryCatch(
    reticulate::py_run_string(
      sprintf(
        "def w(f):\n\tdef _(%s):\n\t\treturn f(%s)\n\treturn _",
        fn_sig, fn_pa
      ),
      local = TRUE, convert = FALSE
    ),
    error = function(e) {
      stop(sprintf("could not convert:\n%s", e$message))
    }
  )

  arg_list <- lapply(stats::setNames(nm = names(fr)), function(x) {
    str2lang(sprintf("reticulate::py_to_r(%s)", x))
  })
  z <- function() {
    do.call(fn, eval(arg_list))
  }
  formals(z) <- fr

  py_str$w(z)
}

#' HiDDEN entrypoint
#'
#' @usage
#' # method for SingleCellExperiment
#' HiDDEN(sce, cond_col, control_val, algo, ...)
#'
#' @usage
#' # method for Seurat
#' HiDDEN(so, cond_col, control_val, algo, ...)
#'
#' @param x input object
#' @param cond_col character
#' @param control_val character
#' @param algo Pipeline
#' @param ... extra arguments passed into pipeline
#'
#' @examplesIf requireNamespace("irlba", quietly = TRUE) && requireNamespace("Seurat", quietly = TRUE) && requireNamespace("SeuratObject", quietly = TRUE) && requireNamespace("SeuratData", quietly = TRUE) && ("ifnb" %in% SeuratData::InstalledData()[["Dataset"]])
#' so <- Seurat::SCTransform(SeuratData::LoadData("ifnb"))
#' sct_pca <- function(so, k) {
#'   irlba::prcomp_irlba(
#'     t(SeuratObject::LayerData(so, assay = "SCT", layer = "scale.data")),
#'     n = k
#'   )$x
#' }
#' algo <- adapters$Pipeline(to_step(sct_pca), methods$log_reg, methods$kmeans_bin)
#' out <- HiDDEN(so, "stim", "CTRL", algo, so = so, k = 15)
#' names(out)
#'
#' @returns HiDDEN output - list of two elements w/ names:
#' - `p_hat`: HiDDEN pipeline per-cell score values
#' - `labs`: HiDDEN-adjusted per-cell condition labels
#'
#' @export
#'
HiDDEN <- S7::new_generic("HiDDEN", "x")

#' @name HiDDEN
#' @method HiDDEN SingleCellExperiment
#'
#' @importClassesFrom SingleCellExperiment SingleCellExperiment
#' @importFrom  SingleCellExperiment SingleCellExperiment
#'
S7::method(HiDDEN, methods::getClass("SingleCellExperiment")) <- function(
    x, cond_col, control_val, algo, ...) {
  HiDDEN(function(e) {
    SingleCellExperiment::colData(x)[[e]]
  }, cond_col, control_val, algo, ...)
}

#' @name HiDDEN
#' @method HiDDEN Seurat
#'
#' @importClassesFrom SeuratObject Seurat
#' @importFrom  SeuratObject CreateSeuratObject
#'
S7::method(HiDDEN, methods::getClass("Seurat")) <- function(
    x, cond_col, control_val, algo, ...) {
  HiDDEN(function(e) {
    x[[e]][, ]
  }, cond_col, control_val, algo, ...)
}

S7::method(HiDDEN, S7::class_function) <- function(
    x, cond_col, control_val, algo, ...) {
  out <- algo(V = x(cond_col) != control_val, ...)

  list(
    "p_hat" = reticulate::py_to_r(out[0]),
    "labs" = ifelse(
      reticulate::py_to_r(out[1]),
      x(cond_col),
      control_val
    )
  )
}

#' HiDDEN entrypoint w/ automatic hyperparameter tuning
#'
#' @usage
#' # method for SingleCellExperiment
#' HiDDENt(sce, cond_col, control_val, algo, tuner, ...)
#'
#' @examplesIf requireNamespace("irlba", quietly = TRUE) && requireNamespace("Seurat", quietly = TRUE) && requireNamespace("SeuratObject", quietly = TRUE) && requireNamespace("SeuratData", quietly = TRUE) && ("ifnb" %in% SeuratData::InstalledData()[["Dataset"]])
#' so <- Seurat::SCTransform(SeuratData::LoadData("ifnb"))
#' sct_pca <- function(so, k) {
#'   irlba::prcomp_irlba(
#'     t(SeuratObject::LayerData(so, assay = "SCT", layer = "scale.data")),
#'     n = k
#'   )$x
#' }
#' algo <- adapters$Pipeline(to_step(sct_pca), methods$log_reg, methods$kmeans_bin)
#' tuner <- tune$FixPointTuner(5, 8, 0.04)
#' out <- HiDDENt(so, "stim", "CTRL", algo, tuner, so = so)
#' names(out)
#' names(out[["outs"]][[as.character(out[["chosen"]])]])
#'
#' @usage
#' # method for Seurat
#' HiDDENt(so, cond_col, control_val, algo, tuner, ...)
#'
#' @param x input object
#' @param cond_col character
#' @param control_val character
#' @param algo Pipeline
#' @param tuner Tuner
#' @param ... extra arguments passed into pipeline
#'
#' @returns HiDDEN tuner output - list of two elements, w/ names:
#' - `chosen`: hyperparameter selected by provided tuner
#' - `outs`: list with names being tested hyper-parameters, values a list of 3 elements, w/ names:
#'    - `p_hat`: HiDDEN pipeline per-cell score values
#'    - `labs`: HiDDEN-adjusted per-cell condition labels
#'    - `score`: tuner score value associated with selected hyper-parameters
#'
#' @export
#'
HiDDENt <- S7::new_generic("HiDDENt", "x")

#' @name HiDDENt
#' @method HiDDEN SingleCellExperiment
#'
#' @importClassesFrom SingleCellExperiment SingleCellExperiment
#' @importFrom  SingleCellExperiment SingleCellExperiment
#'
S7::method(HiDDENt, methods::getClass("SingleCellExperiment")) <- function(
    x, cond_col, control_val, algo, tuner, ...) {
  HiDDENt(function(e) {
    SingleCellExperiment::colData(x)[[e]]
  }, cond_col, control_val, algo, tuner, ...)
}

#' @name HiDDENt
#' @method HiDDEN Seurat
#'
#' @importClassesFrom SeuratObject Seurat
#' @importFrom  SeuratObject CreateSeuratObject
#'
S7::method(HiDDENt, methods::getClass("Seurat")) <- function(
    x, cond_col, control_val, algo, tuner, ...) {
  HiDDENt(function(e) {
    x[[e]][, ]
  }, cond_col, control_val, algo, tuner, ...)
}

S7::method(HiDDENt, S7::class_function) <- function(
    x, cond_col, control_val, algo, tuner, ...) {
  outs <- tuner(algo, V = x(cond_col) != control_val, ...)

  chosen <- reticulate::py_to_r(outs[0])
  out_l <- lapply(reticulate::py_to_r(outs[1]), function(out) {
    list(
      "p_hat" = out[[1]],
      "labs" = ifelse(out[[2]], x(cond_col), control_val),
      "score" = out[[3]]
    )
  })

  names(out_l) <- tryCatch(
    lapply(
      names(out_l),
      function(x) {
        as(x, class(chosen))
      }
    ),
    error = function(e) {
      names(out_l)
    }
  )

  list(
    "chosen" = chosen,
    "outs" = out_l
  )
}
