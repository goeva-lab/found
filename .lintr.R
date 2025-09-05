linters <- linters_with_defaults(
  line_length_linter = line_length_linter(128L),
  object_name_linter = NULL
)

exclusions <- list(
  "R/R/found.R" = list(
    # roxygen2 examplesIf clause can't be multiple lines, so disable line length lints on those lines
    line_length_linter = which(startsWith(readLines("R/R/found.R"), "#' @examplesIf "))
  )
)
