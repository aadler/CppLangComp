\name{CppLangComp-package}
\alias{CppLangComp-package}
\alias{CppLangComp}
\docType{package}
\title{
  C++ versions of layer loss functions for speed comparison
}
\description{
  This package contains various flavors of Rcpp (C++) versions of a simple layer loss cost function. These flavors represent levels of OpenMP parallelism. These functions' performance will be compared with base R and Fortran versions of the same function in the RFortLangComp package. These packages are \bold{not} intended for uploading to CRAN.
}
\details{
  The functions are called by name into the same environment as the functions from RFortLangComp and timings are compared.
}
\author{
Maintainer: Avraham Adler \email{Avraham.Adler@gmail.com}
}
\references{
  There will be a blog post on \url{https://www.avrahamadler.com} that will discuss these packages.
}
\keyword{ package }
\seealso{
  Optional links to other man pages
}
\examples{
  \dontrun{
     ## Optional simple examples of the most important functions
     ## These can be in \dontrun{} and \donttest{} blocks.   
  }
}
