linters <- linters_with_defaults(
  line_length_linter = line_length_linter(128L),
  object_name_linter = NULL
)

exclusions <- list(
  "R/R/found.R" = list(
    line_length_linter = c(61, 133)
  )
)
