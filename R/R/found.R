#' found.adapters
#'
#' adapters module from found python package
#'
#' @export
adapters <- NULL

#' found.methods
#'
#' methods module from found python package
#'
#' @export
methods <- NULL

#' found.tune
#'
#' tune module from found python package
#'
#' @export
tune <- NULL

ad.AnnData <- NULL

#' to_step
#'
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
        fn_sig,
        fn_pa
      ),
      local = TRUE,
      convert = FALSE
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

sce_s4_class <- methods::getClass("SingleCellExperiment", where = "SingleCellExperiment")
seurat_s4_class <- methods::getClass("Seurat", where = "SeuratObject")

conv_HiDDEN <- function(HiDDEN_out) {
  list("p_hat" = c(reticulate::py_to_r(HiDDEN_out[0])), "labs" = c(reticulate::py_to_r(HiDDEN_out[1])))
}

conv_HiDDENt <- function(HiDDENt_outs) {
  chosen <- reticulate::py_to_r(HiDDENt_outs[0])
  out_l <- lapply(
    reticulate::py_to_r(HiDDENt_outs[1]),
    function(out) list("p_hat" = c(out[[1]]), "labs" = c(out[[2]]), "score" = out[[3]])
  )

  names(out_l) <- tryCatch(
    lapply(names(out_l), function(x) as(x, class(chosen))),
    error = function(e) names(out_l)
  )

  list("chosen" = chosen, "outs" = out_l)
}

from_sce <- function(sce, cols) {
  ad.AnnData(
    obs = as.data.frame(SingleCellExperiment::colData(sce)[c(cols)])
  )
}
from_so <- function(so, cols) {
  ad.AnnData(obs = so[[c(cols)]])
}

#' HiDDEN
#'
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
#' character indicating per-cell metadata column for condition labels
#' @param control_val
#' value in per-cell condition labels representing control condition
#' @param algo Pipeline
#' @param ... extra arguments passed into pipeline
#'
#' @examplesIf requireNamespace("irlba", quietly = TRUE) && requireNamespace("Seurat", quietly = TRUE) && requireNamespace("SeuratObject", quietly = TRUE) && requireNamespace("SeuratData", quietly = TRUE) && ("ifnb" %in% SeuratData::InstalledData()[["Dataset"]])
#' so <- SeuratData::LoadData("ifnb") |>
#'   Seurat::SCTransform(verbose = FALSE) |>
#'   suppressWarnings() |>
#'   suppressMessages()
#' sct_pca <- function(so, k) {
#'   irlba::prcomp_irlba(
#'     t(SeuratObject::LayerData(so, assay = "SCT", layer = "scale.data")),
#'     n = k
#'   )$x
#' }
#' algo <- adapters$Pipeline(to_step(sct_pca), methods$reg_logit, methods$bin_kmeans)
#' out <- HiDDEN(so, "stim", "CTRL", algo, so = so, k = 15L)
#' names(out)
#'
#' @returns HiDDEN output - list of two (2) elements w/ names:
#' - `p_hat`: HiDDEN pipeline per-cell score values
#' - `labs`: HiDDEN-adjusted per-cell condition labels
#'
#' @export
#'
HiDDEN <- S7::new_generic("HiDDEN", "x")

#' @name HiDDEN
#' @method HiDDEN SingleCellExperiment
#'
#'
S7::method(HiDDEN, sce_s4_class) <- function(
  x,
  cond_col,
  control_val,
  ...
) {
  HiDDEN(from_sce(x, cond_col), cond_col, control_val, ...)
}

#' @name HiDDEN
#' @method HiDDEN Seurat
#'
#'
S7::method(HiDDEN, seurat_s4_class) <- function(
  x,
  cond_col,
  control_val,
  ...
) {
  HiDDEN(from_so(x, cond_col), cond_col, control_val, ...)
}

#' HiDDENt
#'
#' HiDDEN entrypoint w/ automatic hyperparameter tuning
#'
#' @usage
#' # method for SingleCellExperiment
#' HiDDENt(sce, cond_col, control_val, tuner, algo, ...)
#'
#' @examplesIf requireNamespace("irlba", quietly = TRUE) && requireNamespace("Seurat", quietly = TRUE) && requireNamespace("SeuratObject", quietly = TRUE) && requireNamespace("SeuratData", quietly = TRUE) && ("ifnb" %in% SeuratData::InstalledData()[["Dataset"]])
#' so <- SeuratData::LoadData("ifnb") |>
#'   Seurat::SCTransform(verbose = FALSE) |>
#'   suppressWarnings() |>
#'   suppressMessages()
#' sct_pca <- function(so, k) {
#'   irlba::prcomp_irlba(
#'     t(SeuratObject::LayerData(so, assay = "SCT", layer = "scale.data")),
#'     n = k
#'   )$x
#' }
#' algo <- adapters$Pipeline(to_step(sct_pca), methods$reg_logit, methods$bin_kmeans)
#' tuner <- tune$FixPointTuner(5, 8, 0.04)
#' out <- HiDDENt(so, "stim", "CTRL", tuner, algo, so = so)
#' names(out)
#' names(out[["outs"]][[as.character(out[["chosen"]])]])
#'
#' @usage
#' # method for Seurat
#' HiDDENt(so, cond_col, control_val, tuner, algo, ...)
#'
#' @param x input object
#' @param cond_col character
#' character indicating per-cell metadata column for condition labels
#' @param control_val
#' value in per-cell condition labels representing control condition
#' @param tuner Tuner
#' @param algo Pipeline
#' @param ... extra arguments passed into pipeline
#'
#' @returns HiDDENt output - list of two (2) elements, w/ names:
#' - `chosen`: hyperparameter selected by provided tuner
#' - `outs`: list of three (3) elements w/ names:
#'    - `p_hat`: HiDDEN pipeline per-cell score values
#'    - `labs`: HiDDEN-adjusted per-cell condition labels
#'    - `score`: tuner score value associated with selected hyper-parameters
#'
#' @export
#'
HiDDENt <- S7::new_generic("HiDDENt", "x")

#' @name HiDDENt
#' @method HiDDENt SingleCellExperiment
#'
#'
S7::method(HiDDENt, sce_s4_class) <- function(
  x,
  cond_col,
  control_val,
  ...
) {
  HiDDENt(from_sce(x, cond_col), cond_col, control_val, ...)
}

#' @name HiDDENt
#' @method HiDDENt Seurat
#'
#'
S7::method(HiDDENt, seurat_s4_class) <- function(
  x,
  cond_col,
  control_val,
  ...
) {
  HiDDENt(from_so(x, cond_col), cond_col, control_val, ...)
}

#' HiDDENg
#'
#' HiDDEN entrypoint w/ grouping
#'
#' @usage
#' # method for SingleCellExperiment
#' HiDDENg(sce, cond_col, control_val, group_by, algo, which_grouped, grp_specific_args, ...)
#'
#' @usage
#' # method for Seurat
#' HiDDENg(so, cond_col, control_val, group_by, algo, which_grouped, grp_specific_args, ...)
#'
#' @param x input object
#' @param cond_col character
#' character indicating per-cell metadata column for condition labels
#' @param control_val
#' value in per-cell condition labels representing control condition
#' @param group_by character
#' character indicating per-cell metadata column for grouping labels
#' @param algo Pipeline
#' @param which_grouped character
#' vector of argument names from the pipeline need to be grouped
#' (by default will be determined automatically by checking which ones support indexing)
#' @param grp_specific_args list
#' any arguments that need to be provided on a group-specific basis
#' (list w/ names being groups, values being lists w/ names being arguments)
#' (by default will be determined automatically by checking which ones support indexing)
#' @param ... extra arguments passed into pipeline
#'
#' @examplesIf requireNamespace("irlba", quietly = TRUE) && requireNamespace("Seurat", quietly = TRUE) && requireNamespace("SeuratObject", quietly = TRUE) && requireNamespace("SeuratData", quietly = TRUE) && ("ifnb" %in% SeuratData::InstalledData()[["Dataset"]])
#' so <- SeuratData::LoadData("ifnb") |>
#'   Seurat::SCTransform(verbose = FALSE) |>
#'   suppressWarnings() |>
#'   suppressMessages()
#' sct_pca <- function(so, k) {
#'   irlba::prcomp_irlba(
#'     t(SeuratObject::LayerData(so, assay = "SCT", layer = "scale.data")),
#'     n = k
#'   )$x
#' }
#' algo <- adapters$Pipeline(to_step(sct_pca), methods$reg_logit, methods$bin_kmeans)
#' out <- HiDDENg(so, "stim", "CTRL", "seurat_annotations", algo, so = so, k = 15L)
#' names(out)
#'
#' @returns HiDDEN output - list of two (2) elements w/ names:
#' - `p_hat`: HiDDEN pipeline per-cell score values
#' - `labs`: HiDDEN-adjusted per-cell condition labels
#'
#' @export
#'
HiDDENg <- S7::new_generic("HiDDENg", "x")

#' @name HiDDENg
#' @method HiDDENg SingleCellExperiment
#'
#'
S7::method(HiDDENg, sce_s4_class) <- function(
  x,
  cond_col,
  control_val,
  group_by,
  ...
) {
  HiDDENg(from_sce(x, c(cond_col, group_by)), cond_col, control_val, group_by, ...)
}
#' @name HiDDENg
#' @method HiDDENg Seurat
#'
#'
S7::method(HiDDENg, seurat_s4_class) <- function(
  x,
  cond_col,
  control_val,
  group_by,
  ...
) {
  HiDDENg(from_so(x, c(cond_col, group_by)), cond_col, control_val, group_by, ...)
}

#' HiDDENgt
#'
#' HiDDEN entrypoint w/ grouping and automatic hyper-parameter tuning
#'
#' @usage
#' # method for SingleCellExperiment
#' HiDDENgt(sce, cond_col, control_val, group_by, tuner, algo, which_grouped, grp_specific_args, ...)
#'
#' @usage
#' # method for Seurat
#' HiDDENgt(so, cond_col, control_val, group_by, tuner, algo, which_grouped, grp_specific_args, ...)
#'
#' @param x input object
#' @param cond_col character
#' character indicating per-cell metadata column for condition labels
#' @param control_val
#' value in per-cell condition labels representing control condition
#' @param group_by character
#' character indicating per-cell metadata column for grouping labels
#' @param tuner Tuner
#' @param algo Pipeline
#' @param which_grouped character
#' vector of argument names from the pipeline need to be grouped
#' (by default will be determined automatically by checking which ones support indexing)
#' @param grp_specific_args list
#' any arguments that need to be provided on a group-specific basis
#' (list w/ names being groups, values being lists w/ names being arguments names, values being associated value)
#' (by default will be determined automatically by checking which ones support indexing)
#' @param ... extra arguments passed into pipeline
#'
#' @examplesIf requireNamespace("irlba", quietly = TRUE) && requireNamespace("Seurat", quietly = TRUE) && requireNamespace("SeuratObject", quietly = TRUE) && requireNamespace("SeuratData", quietly = TRUE) && ("ifnb" %in% SeuratData::InstalledData()[["Dataset"]])
#' so <- SeuratData::LoadData("ifnb") |>
#'   Seurat::SCTransform(verbose = FALSE) |>
#'   suppressWarnings() |>
#'   suppressMessages()
#' sct_pca <- function(so, k) {
#'   irlba::prcomp_irlba(
#'     t(SeuratObject::LayerData(so, assay = "SCT", layer = "scale.data")),
#'     n = k
#'   )$x
#' }
#' algo <- adapters$Pipeline(to_step(sct_pca), methods$reg_logit, methods$bin_kmeans)
#' tuner <- tune$FixPointTuner(5, 8, 0.04)
#' out <- HiDDENgt(so, "stim", "CTRL", "seurat_annotations", tuner, algo, so = so)
#' names(out)
#' names(out[["by_param"]](out[["chosen"]]))
#' names(out[["by_grp"]](names(out[["chosen"]])[[1]]))
#'
#' @returns HiDDENgt output - list of three (3) elements w/ names:
#' - `chosen`: mapping from each group to selected hyper-parameter for that group
#' - `by_param`: accessor function which given a mapping of groups to hyper-parameters
#' (and an optional default hyper-parameter for unspecified groups), returns a list of three (3) elements w/ names:
#'    - `p_hat`: 1-d array of prediction outputs by model, ordered by their original order within the provided
#'    - `labs`: model adjusted labels
#'    - `score`: mapping of group to score value given provided configuration
#' - `by_grp`: accessor function which given a specific group, returns HiDDENt-style output for just that group
#' @export
#'
HiDDENgt <- S7::new_generic("HiDDENgt", "x")

#' @name HiDDENgt
#' @method HiDDENgt SingleCellExperiment
#'
#'
S7::method(HiDDENgt, sce_s4_class) <- function(
  x,
  cond_col,
  control_val,
  group_by,
  ...
) {
  HiDDENgt(from_sce(x, c(cond_col, group_by)), cond_col, control_val, group_by, ...)
}

#' @name HiDDENgt
#' @method HiDDENgt Seurat
#'
#'
S7::method(HiDDENgt, seurat_s4_class) <- function(
  x,
  cond_col,
  control_val,
  group_by,
  ...
) {
  HiDDENgt(from_so(x, c(cond_col, group_by)), cond_col, control_val, group_by, ...)
}

.onLoad <- function(libname, pkgname) {
  reticulate::py_require(c("anndata", "found"))

  ad.AnnData <<- reticulate::import("anndata", delay_load = TRUE, convert = FALSE)$AnnData
  found.find <- reticulate::import("found.find", delay_load = TRUE, convert = FALSE)

  adapters <<- reticulate::import("found.adapters", delay_load = TRUE, convert = FALSE)
  methods <<- reticulate::import("found.methods", delay_load = TRUE, convert = FALSE)
  tune <<- reticulate::import("found.tune", delay_load = TRUE, convert = FALSE)

  anndata_s3_class <- S7::new_S3_class(nameOfClass(ad.AnnData))

  S7::method(HiDDEN, anndata_s3_class) <- function(x, cond_col, control_val, ...) {
    conv_HiDDEN(found.find$HiDDEN(x, cond_col, control_val, ...))
  }

  S7::method(HiDDENt, anndata_s3_class) <- function(x, cond_col, control_val, ...) {
    conv_HiDDENt(found.find$HiDDENt(x, cond_col, control_val, ...))
  }

  # try to automatically detect case where so/sce object is passed to kwargs, and add which_grouped handler for this
  gwrap_auto <- function(fn, x, cond_col, control_val, group_by, ...) {
    togroup_idx <- names(Filter(
      function(e) {
        (class(e)[[1]] %in% c("SingleCellExperiment", "Seurat")) && reticulate::py_to_r(ncol(e) == x$n_obs)
      },
      list(...)
    ))
    if (length(togroup_idx) > 0) {
      which_grouped <- list()
      for (i in togroup_idx) {
        which_grouped[[i]] <- to_step(function(e, idx) e[, idx + 1])
      }
      fn(x, cond_col, control_val, group_by, which_grouped = which_grouped, ...)
    } else {
      fn(x, cond_col, control_val, group_by, ...)
    }
  }

  S7::method(HiDDENg, anndata_s3_class) <- function(x, cond_col, control_val, group_by, ...) {
    conv_HiDDEN(gwrap_auto(found.find$HiDDENg, x, cond_col, control_val, group_by, ...))
  }

  S7::method(HiDDENgt, anndata_s3_class) <- function(x, cond_col, control_val, group_by, ...) {
    outs <- gwrap_auto(found.find$HiDDENgt, x, cond_col, control_val, group_by, ...)

    list(
      "chosen" = reticulate::py_to_r(outs[0]),
      "by_param" = function(mapping = NULL, default = NULL) {
        out <- outs[1](if (is.null(mapping)) NULL else reticulate::dict(mapping), default)
        append(conv_HiDDEN(reticulate::tuple(out[0], out[1])), list("score" = reticulate::py_to_r(out[2])))
      },
      "by_grp" = function(grp) conv_HiDDENt(reticulate::tuple(outs[0][grp], outs[2](grp)))[["outs"]]
    )
  }

  S7::methods_register()
}
