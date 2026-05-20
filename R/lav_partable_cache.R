# store pta in attributes of partable
lav_partable_set_cache <- function(partable, pta = NULL, force = FALSE) {
  if (!force &&
      !is.null(attr(partable, "vnames", exact = TRUE)) &&
      !is.null(attr(partable, "nvar", exact = TRUE))) {
    return(partable)                    # cache already OK
  }
  if (is.null(pta)) {
    if (force) attr(partable, "vnames") <- NULL
    pta <- lav_partable_attributes(partable)
  }

  partable_attributes <- attributes(partable)
  partable_attributes[names(pta)] <- pta
  attributes(partable) <- partable_attributes

  partable
}

lav_partable_remove_cache <- function(partable) {
  partable_attributes <- attributes(partable)
  keep <- names(partable_attributes) %in% c("ovda", "names")
  attributes(partable) <- partable_attributes[keep]

  partable
}
