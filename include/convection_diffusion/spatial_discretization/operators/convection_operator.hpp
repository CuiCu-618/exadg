#include "convection_operator.h"

#include <navier-constants.h>

namespace ConvDiff
{

#if DIM_2 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 1, float>;
#endif
#if DIM_2 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 1, double>;
#endif

#if DIM_2 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 2, float>;
#endif
#if DIM_2 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 2, double>;
#endif

#if DIM_2 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 3, float>;
#endif
#if DIM_2 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 3, double>;
#endif

#if DIM_2 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 4, float>;
#endif
#if DIM_2 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 4, double>;
#endif

#if DIM_2 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 5, float>;
#endif
#if DIM_2 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 5, double>;
#endif

#if DIM_2 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 6, float>;
#endif
#if DIM_2 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 6, double>;
#endif

#if DIM_2 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 7, float>;
#endif
#if DIM_2 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 7, double>;
#endif

#if DIM_2 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 8, float>;
#endif
#if DIM_2 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 8, double>;
#endif

#if DIM_2 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<2, 9, float>;
#endif
#if DIM_2 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<2, 9, double>;
#endif

#if DIM_3 && DEGREE_1 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 1, float>;
#endif
#if DIM_3 && DEGREE_1 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 1, double>;
#endif

#if DIM_3 && DEGREE_2 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 2, float>;
#endif
#if DIM_3 && DEGREE_2 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 2, double>;
#endif

#if DIM_3 && DEGREE_3 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 3, float>;
#endif
#if DIM_3 && DEGREE_3 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 3, double>;
#endif

#if DIM_3 && DEGREE_4 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 4, float>;
#endif
#if DIM_3 && DEGREE_4 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 4, double>;
#endif

#if DIM_3 && DEGREE_5 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 5, float>;
#endif
#if DIM_3 && DEGREE_5 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 5, double>;
#endif

#if DIM_3 && DEGREE_6 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 6, float>;
#endif
#if DIM_3 && DEGREE_6 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 6, double>;
#endif

#if DIM_3 && DEGREE_7 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 7, float>;
#endif
#if DIM_3 && DEGREE_7 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 7, double>;
#endif

#if DIM_3 && DEGREE_8 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 8, float>;
#endif
#if DIM_3 && DEGREE_8 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 8, double>;
#endif

#if DIM_3 && DEGREE_9 && (OP_FLOAT || MG_FLOAT)
template class ConvectiveOperator<3, 9, float>;
#endif
#if DIM_3 && DEGREE_9 && (OP_DOUBLE || MG_DOUBLE)
template class ConvectiveOperator<3, 9, double>;
#endif

}

