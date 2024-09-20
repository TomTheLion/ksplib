#include "rk853.h"

namespace rk853
{
	// Machine precision
	constexpr double eps = 2.220446049250313E-016;

	// Node coefficients
	constexpr double c2 = 0.526001519587677318785587544488e-01;
	constexpr double c3 = 0.789002279381515978178381316732e-01;
	constexpr double c4 = 0.118350341907227396726757197510e+00;
	constexpr double c5 = 0.281649658092772603273242802490e+00;
	constexpr double c6 = 0.333333333333333333333333333333e+00;
	constexpr double c7 = 0.25e+00;
	constexpr double c8 = 0.307692307692307692307692307692e+00;
	constexpr double c9 = 0.651282051282051282051282051282e+00;
	constexpr double c10 = 0.6e+00;
	constexpr double c11 = 0.857142857142857142857142857142e+00;
	constexpr double c14 = 0.1e+00;
	constexpr double c15 = 0.2e+00;
	constexpr double c16 = 0.777777777777777777777777777778e+00;

	// Runge-Kutta matrix
	constexpr double a21 = 5.26001519587677318785587544488e-2;
	constexpr double a31 = 1.97250569845378994544595329183e-2;
	constexpr double a32 = 5.91751709536136983633785987549e-2;
	constexpr double a41 = 2.95875854768068491816892993775e-2;
	constexpr double a43 = 8.87627564304205475450678981324e-2;
	constexpr double a51 = 2.41365134159266685502369798665e-1;
	constexpr double a53 = -8.84549479328286085344864962717e-1;
	constexpr double a54 = 9.24834003261792003115737966543e-1;
	constexpr double a61 = 3.7037037037037037037037037037e-2;
	constexpr double a64 = 1.70828608729473871279604482173e-1;
	constexpr double a65 = 1.25467687566822425016691814123e-1;
	constexpr double a71 = 3.7109375e-2;
	constexpr double a74 = 1.70252211019544039314978060272e-1;
	constexpr double a75 = 6.02165389804559606850219397283e-2;
	constexpr double a76 = -1.7578125e-2;
	constexpr double a81 = 3.70920001185047927108779319836e-2;
	constexpr double a84 = 1.70383925712239993810214054705e-1;
	constexpr double a85 = 1.07262030446373284651809199168e-1;
	constexpr double a86 = -1.53194377486244017527936158236e-2;
	constexpr double a87 = 8.27378916381402288758473766002e-3;
	constexpr double a91 = 6.24110958716075717114429577812e-1;
	constexpr double a94 = -3.36089262944694129406857109825e0;
	constexpr double a95 = -8.68219346841726006818189891453e-1;
	constexpr double a96 = 2.75920996994467083049415600797e1;
	constexpr double a97 = 2.01540675504778934086186788979e1;
	constexpr double a98 = -4.34898841810699588477366255144e1;
	constexpr double a101 = 4.77662536438264365890433908527e-1;
	constexpr double a104 = -2.48811461997166764192642586468e0;
	constexpr double a105 = -5.90290826836842996371446475743e-1;
	constexpr double a106 = 2.12300514481811942347288949897e1;
	constexpr double a107 = 1.52792336328824235832596922938e1;
	constexpr double a108 = -3.32882109689848629194453265587e1;
	constexpr double a109 = -2.03312017085086261358222928593e-2;
	constexpr double a111 = -9.3714243008598732571704021658e-1;
	constexpr double a114 = 5.18637242884406370830023853209e0;
	constexpr double a115 = 1.09143734899672957818500254654e0;
	constexpr double a116 = -8.14978701074692612513997267357e0;
	constexpr double a117 = -1.85200656599969598641566180701e1;
	constexpr double a118 = 2.27394870993505042818970056734e1;
	constexpr double a119 = 2.49360555267965238987089396762e0;
	constexpr double a1110 = -3.0467644718982195003823669022e0;
	constexpr double a121 = 2.27331014751653820792359768449e0;
	constexpr double a124 = -1.05344954667372501984066689879e1;
	constexpr double a125 = -2.00087205822486249909675718444e0;
	constexpr double a126 = -1.79589318631187989172765950534e1;
	constexpr double a127 = 2.79488845294199600508499808837e1;
	constexpr double a128 = -2.85899827713502369474065508674e0;
	constexpr double a129 = -8.87285693353062954433549289258e0;
	constexpr double a1210 = 1.23605671757943030647266201528e1;
	constexpr double a1211 = 6.43392746015763530355970484046e-1;

	// Additional coefficients for dense output
	constexpr double a141 = 5.61675022830479523392909219681e-2;
	constexpr double a147 = 2.53500210216624811088794765333e-1;
	constexpr double a148 = -2.46239037470802489917441475441e-1;
	constexpr double a149 = -1.24191423263816360469010140626e-1;
	constexpr double a1410 = 1.5329179827876569731206322685e-1;
	constexpr double a1411 = 8.20105229563468988491666602057e-3;
	constexpr double a1412 = 7.56789766054569976138603589584e-3;
	constexpr double a1413 = -8.298e-3;
	constexpr double a151 = 3.18346481635021405060768473261e-2;
	constexpr double a156 = 2.83009096723667755288322961402e-2;
	constexpr double a157 = 5.35419883074385676223797384372e-2;
	constexpr double a158 = -5.49237485713909884646569340306e-2;
	constexpr double a1511 = -1.08347328697249322858509316994e-4;
	constexpr double a1512 = 3.82571090835658412954920192323e-4;
	constexpr double a1513 = -3.40465008687404560802977114492e-4;
	constexpr double a1514 = 1.41312443674632500278074618366e-1;
	constexpr double a161 = -4.28896301583791923408573538692e-1;
	constexpr double a166 = -4.69762141536116384314449447206e0;
	constexpr double a167 = 7.68342119606259904184240953878e0;
	constexpr double a168 = 4.06898981839711007970213554331e0;
	constexpr double a169 = 3.56727187455281109270669543021e-1;
	constexpr double a1613 = -1.39902416515901462129418009734e-3;
	constexpr double a1614 = 2.9475147891527723389556272149e0;
	constexpr double a1615 = -9.15095847217987001081870187138e0;


	// Weight coefficients
	constexpr double b1 = 5.42937341165687622380535766363e-2;
	constexpr double b6 = 4.45031289275240888144113950566e0;
	constexpr double b7 = 1.89151789931450038304281599044e0;
	constexpr double b8 = -5.8012039600105847814672114227e0;
	constexpr double b9 = 3.1116436695781989440891606237e-1;
	constexpr double b10 = -1.52160949662516078556178806805e-1;
	constexpr double b11 = 2.01365400804030348374776537501e-1;
	constexpr double b12 = 4.47106157277725905176885569043e-2;

	// Error 3 coeffients
	constexpr double e31 = 0.244094488188976377952755905512e+00;
	constexpr double e32 = 0.733846688281611857341361741547e+00;
	constexpr double e33 = 0.220588235294117647058823529412e-01;

	// Error 5 coefficients
	constexpr double e51 = 0.1312004499419488073250102996e-01;
	constexpr double e56 = -0.1225156446376204440720569753e+01;
	constexpr double e57 = -0.4957589496572501915214079952e+00;
	constexpr double e58 = 0.1664377182454986536961530415e+01;
	constexpr double e59 = -0.3503288487499736816886487290e+00;
	constexpr double e510 = 0.3341791187130174790297318841e+00;
	constexpr double e511 = 0.8192320648511571246570742613e-01;
	constexpr double e512 = -0.2235530786388629525884427845e-01;

	// Interpolation coefficients
	constexpr double d41 = -0.84289382761090128651353491142e+01;
	constexpr double d46 = 0.56671495351937776962531783590e+00;
	constexpr double d47 = -0.30689499459498916912797304727e+01;
	constexpr double d48 = 0.23846676565120698287728149680e+01;
	constexpr double d49 = 0.21170345824450282767155149946e+01;
	constexpr double d410 = -0.87139158377797299206789907490e+00;
	constexpr double d411 = 0.22404374302607882758541771650e+01;
	constexpr double d412 = 0.63157877876946881815570249290e+00;
	constexpr double d413 = -0.88990336451333310820698117400e-01;
	constexpr double d414 = 0.18148505520854727256656404962e+02;
	constexpr double d415 = -0.91946323924783554000451984436e+01;
	constexpr double d416 = -0.44360363875948939664310572000e+01;
	constexpr double d51 = 0.10427508642579134603413151009e+02;
	constexpr double d56 = 0.24228349177525818288430175319e+03;
	constexpr double d57 = 0.16520045171727028198505394887e+03;
	constexpr double d58 = -0.37454675472269020279518312152e+03;
	constexpr double d59 = -0.22113666853125306036270938578e+02;
	constexpr double d510 = 0.77334326684722638389603898808e+01;
	constexpr double d511 = -0.30674084731089398182061213626e+02;
	constexpr double d512 = -0.93321305264302278729567221706e+01;
	constexpr double d513 = 0.15697238121770843886131091075e+02;
	constexpr double d514 = -0.31139403219565177677282850411e+02;
	constexpr double d515 = -0.93529243588444783865713862664e+01;
	constexpr double d516 = 0.35816841486394083752465898540e+02;
	constexpr double d61 = 0.19985053242002433820987653617e+02;
	constexpr double d66 = -0.38703730874935176555105901742e+03;
	constexpr double d67 = -0.18917813819516756882830838328e+03;
	constexpr double d68 = 0.52780815920542364900561016686e+03;
	constexpr double d69 = -0.11573902539959630126141871134e+02;
	constexpr double d610 = 0.68812326946963000169666922661e+01;
	constexpr double d611 = -0.10006050966910838403183860980e+01;
	constexpr double d612 = 0.77771377980534432092869265740e+00;
	constexpr double d613 = -0.27782057523535084065932004339e+01;
	constexpr double d614 = -0.60196695231264120758267380846e+02;
	constexpr double d615 = 0.84320405506677161018159903784e+02;
	constexpr double d616 = 0.11992291136182789328035130030e+02;
	constexpr double d71 = -0.25693933462703749003312586129e+02;
	constexpr double d76 = -0.15418974869023643374053993627e+03;
	constexpr double d77 = -0.23152937917604549567536039109e+03;
	constexpr double d78 = 0.35763911791061412378285349910e+03;
	constexpr double d79 = 0.93405324183624310003907691704e+02;
	constexpr double d710 = -0.37458323136451633156875139351e+02;
	constexpr double d711 = 0.10409964950896230045147246184e+03;
	constexpr double d712 = 0.29840293426660503123344363579e+02;
	constexpr double d713 = -0.43533456590011143754432175058e+02;
	constexpr double d714 = 0.96324553959188282948394950600e+02;
	constexpr double d715 = -0.39177261675615439165231486172e+02;
	constexpr double d716 = -0.14972683625798562581422125276e+03;

	// Tests if a value falls within a range
	// a = value to be tested
	// r1, r2 = bounds of the range
	bool in_range(double a, double r1, double r2)
	{
		return r1 <= r2 ? r1 <= a && a <= r2 : r2 <= a && a <= r1;
	}

	// Initializes Equation
	// f = pointer to function that calculates the derivative of the problem
	// iflag = flag which holds the return status of the integrator
	// neqn = number of equations
	// t = initial time
	// y = initial state of the problem
	// yp = derivative of problem
	// iwork = internal work space for the integrator
	// work = internal work space for the integrator
	// params = optional pointer to parameters for f
	void init(
		void f(double t, double y[], double yp[], void* params),
		int& iflag,
		int neqn,
		double t,
		std::vector<double>& y,
		std::vector<double>& yp,
		std::vector<int>& iwork,
		std::vector<double>& work,
		void* params)
	{
		// iflag of 1 indicates integrator is in its initial state
		iflag = 1;
		// set the size of yp and the internal workspace
		yp.resize(neqn);
		iwork.resize(1);
		work.resize(5 + 22 * neqn);
		// set the initial value of errold, t, and internal copies of the problem state
		// and its derivative
		double& errold = work[0];
		double& tt = work[3];
		double* yy = &work[5];
		double* yyp = yy + neqn;
		errold = 1.0;
		tt = t;
		f(t, y.data(), yp.data(), params);
		for (int i = 0; i < neqn; i++)
		{
			yy[i] = y[i];
			yyp[i] = yp[i];
		}
	}

	// Steps Equation from t to tout
	// f = pointer to function that calculates the derivative of the problem
	// max_iter = maximum number of iterations
	// iflag = flag which holds the return status of the integrator
	// neqn = number of equations
	// reltol = relative error tolerance
	// abstol = absolute error tolerance
	// t = initial time
	// tout = final desired integration time
	// y = initial state of the problem
	// yp = derivative of problem
	// iwork = internal work space for the integrator
	// work = internal work space for the integrator
	// params = optional pointer to parameters for f
	void step(
		void f(double t, double y[], double yp[], void* params),
		int max_iter,
		int& tot_iter,
		int& rej_iter,
		int& iflag,
		int neqn,
		double reltol,
		double abstol,
		double& t,
		double tout,
		std::vector<double>& y,
		std::vector<double>& yp,
		std::vector<int>& iwork,
		std::vector<double>& work,
		void* params)
	{
		int& reject = iwork[0];

		double& errold = work[0];
		double& d = work[1];
		double& hh = work[2];
		double& tt = work[3];
		double& tw = work[4];
		double tti = tt;

		double* yy = &work[5];
		double* yyp = yy + neqn;
		double* yw = yyp + neqn;
		double* yw2 = yw + neqn;
		double* ywp = yw2 + neqn;
		double* k2 = ywp + neqn;
		double* k3 = k2 + neqn;
		double* k4 = k3 + neqn;
		double* k5 = k4 + neqn;
		double* k6 = k5 + neqn;
		double* k7 = k6 + neqn;
		double* k8 = k7 + neqn;
		double* k9 = k8 + neqn;
		double* k10 = k9 + neqn;
		double* r1 = k10 + neqn;
		double* r2 = r1 + neqn;
		double* r3 = r2 + neqn;
		double* r4 = r3 + neqn;
		double* r5 = r4 + neqn;
		double* r6 = r5 + neqn;
		double* r7 = r6 + neqn;
		double* r8 = r7 + neqn;

		// if integrator is in initial state, set initial stepsize
		// if integrator was previously successful and tout falls within the last
		// step interpolate and return
		if (iflag == 1)
		{
			initial_step_size(neqn, reltol, abstol, hh, yy, yyp);
		}
		else if (iflag == 2 && in_range(tout, tt, tw))
		{
			intrp(neqn, t, tout, y, yp, tt, tw, r1, r2, r3, r4, r5, r6, r7, r8);
			return;
		}

		// set direction of integration
		d = tout >= tt ? 1.0 : -1.0;

		// main integration loop
		for (int iter = 0; iter < max_iter; iter++)
		{
			tot_iter++;

			if (hh < 4.0 * eps * abs(tt - tti))
			{
				// iflag of 3 indicates that the error tolerances are too low
				iflag = 3;
			}

			// perform step then estimate error and update step size
			dy(f, neqn, d * hh, tt, yy, yyp, yw, k2, k3, k4, k5, k6, k7, k8, k9, k10, params);
			update_step_size(neqn, reltol, abstol, reject, errold, d, hh, tt, tw, yy, yyp, yw, k2, k3, k4, k5, k6, k7, k8, k9, k10);

			// test if step was successful
			if (reject)
			{
				rej_iter++;
				continue;
			}

			// if step was successful and tout falls within the step prepare dense
			// output, interpolate, and return, if step was successful and tout falls
			// outside the step prepare the next step
			if (in_range(tout, tt, tw))
			{
				// iflag of 2 indicates that integration was successful
				iflag = 2;
				f(tw, yw, ywp, params);
				dense(f, neqn, tt, tw, yy, yyp, yw, yw2, ywp, k2, k3, k4, k5, k6, k7, k8, k9, k10, r1, r2, r3, r4, r5, r6, r7, r8, params);
				double temp = tt;
				tt = tw;
				tw = temp;
				for (int i = 0; i < neqn; i++)
				{
					yy[i] = yw[i];
					yyp[i] = ywp[i];
				}
				intrp(neqn, t, tout, y, yp, tt, tw, r1, r2, r3, r4, r5, r6, r7, r8);
				return;
			}
			else
			{
				f(tw, yw, ywp, params);
				tt = tw;
				for (int i = 0; i < neqn; i++)
				{
					yy[i] = yw[i];
					yyp[i] = ywp[i];
				}
			}
		}

		// iflag of 4 indicates that the maximum number of iterations was exceeded
		iflag = 4;
	}

	// Calculates Runge-Kutta steps
	// f = pointer to function that calculates the derivative of the problem
	// neqn = number of equations
	// h = current step size with direction
	// tt = initial time
	// yy = current state of the problem
	// yyp = current derivative of problem
	// yw = working state of the problem
	// k = Runge-kutta slope estimates
	// params = optional pointer to parameters for f
	void dy(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		double h,
		double tt,
		double* yy,
		double* yyp,
		double* yw,
		double* k2,
		double* k3,
		double* k4,
		double* k5,
		double* k6,
		double* k7,
		double* k8,
		double* k9,
		double* k10,
		void* params)
	{
		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a21 * yyp[i]);
		f(tt + c2 * h, yw, k2, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a31 * yyp[i] + a32 * k2[i]);
		f(tt + c3 * h, yw, k3, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a41 * yyp[i] + a43 * k3[i]);
		f(tt + c4 * h, yw, k4, params);
	
		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a51 * yyp[i] + a53 * k3[i] + a54 * k4[i]);
		f(tt + c5 * h, yw, k5, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a61 * yyp[i] + a64 * k4[i] + a65 * k5[i]);
		f(tt + c6 * h, yw, k6, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a71 * yyp[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
		f(tt + c7 * h, yw, k7, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a81 * yyp[i] + a84 * k4[i] + a85 * k5[i] + a86 * k6[i] + a87 * k7[i]);
		f(tt + c8 * h, yw, k8, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a91 * yyp[i] + a94 * k4[i] + a95 * k5[i] + a96 * k6[i] + a97 * k7[i] + a98 * k8[i]);
		f(tt + c9 * h, yw, k9, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a101 * yyp[i] + a104 * k4[i] + a105 * k5[i] + a106 * k6[i] + a107 * k7[i] + a108 * k8[i] + a109 * k9[i]);
		f(tt + c10 * h, yw, k10, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a111 * yyp[i] + a114 * k4[i] + a115 * k5[i] + a116 * k6[i] + a117 * k7[i] + a118 * k8[i] + a119 * k9[i] + a1110 * k10[i]);
		f(tt + c11 * h, yw, k2, params);
		
		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a121 * yyp[i] + a124 * k4[i] + a125 * k5[i] + a126 * k6[i] + a127 * k7[i] + a128 * k8[i] + a129 * k9[i] + a1210 * k10[i] + a1211 * k2[i]);
		f(tt + h, yw, k3, params);

		for (int i = 0; i < neqn; i++)
		{
			k4[i] = b1 * yyp[i] + b6 * k6[i] + b7 * k7[i] + b8 * k8[i] + b9 * k9[i] + b10 * k10[i] + b11 * k2[i] + b12 * k3[i];
			yw[i] = yy[i] + h * k4[i];
		}
	}

	// Estimates the initial step size
	// neqn = number of equations
	// reltol = relative error tolerance
	// abstol = absolute error tolerance
	// hh = current step size
	// yy = current state of the problem
	// yyp = current derivative of problem
	void initial_step_size(
		int neqn,
		double reltol,
		double abstol,
		double& hh,
		double* yy,
		double* yyp)
	{
		double err = 0.0;

		for (int i = 0; i < neqn; i++)
		{
			double sci = abstol + reltol * abs(yy[i]);
			double ei = yyp[i];
			err += pow(ei / sci, 2.0);
		}

		hh = pow(err / neqn, -0.0625);
	}

	// Estimates error and calculates next step size
	// neqn = number of equations
	// reltol = relative error tolerance
	// abstol = absolute error tolerance
	// reject = indicates if step was rejected
	// errold = err from previous step
	// d = direction of integration
	// hh = current step size
	// tt = initial time
	// tw = working time
	// yy = current state of the problem
	// yyp = current derivative of problem
	// yw = working state of the problem
	// k = Runge-kutta slope estimates
	void update_step_size(
		int neqn,
		double reltol,
		double abstol,
		int& reject,
		double& errold,
		double d,
		double& hh,
		double tt,
		double& tw,
		double* yy,
		double* yyp,
		double* yw,
		double* k2,
		double* k3,
		double* k4,
		double* k5,
		double* k6,
		double* k7,
		double* k8,
		double* k9,
		double* k10)
	{
		// step size control parameters
		const double alpha = 0.125;
		const double beta = 0.0;
		const double safe = 0.9;
		const double min_scale = 0.333;
		const double max_scale = 6.0;

		// esimate error
		double scale;
		double err3 = 0.0;
		double err5 = 0.0;

		for (int i = 0; i < neqn; i++)
		{
			double sci = abstol + reltol * std::max(abs(yy[i]), abs(yw[i]));
			double e3i = k4[i] - e31 * yyp[i] - e32 * k9[i] - e33 * k3[i];
			double e5i = e51 * yyp[i] + e56 * k6[i] + e57 * k7[i] + e58 * k8[i] + e59 * k9[i] + e510 * k10[i] + e511 * k2[i] + e512 * k3[i];
			err3 += pow(e3i / sci, 2.0);
			err5 += pow(e5i / sci, 2.0);
		}

		double denom = err5 + 0.01 * err3;	
		double err = denom > 0 ? hh * err5 / sqrt(1.0 / (neqn * denom)) : 0.0;

		// if err < 1.0 accept step, update tw, hh, errold, and set reject to 0
		// if err > 1.0 reject step, update hh, and set reject to 1
		if (err < 1.0)
		{
			if (err == 0.0)
			{
				scale = max_scale;
			}
			else
			{
				scale = safe * pow(err, -alpha) * pow(errold, beta);
				if (scale < min_scale) { scale = min_scale; }
				if (scale > max_scale) { scale = max_scale; }
			}
			if (reject)
			{
				tw = tt + d * hh;
				hh *= std::min(scale, 1.0);
			}
			else
			{
				tw = tt + d * hh;
				hh *= scale;
			}
			errold = std::max(err, 1e-4);
			reject = 0;
		}
		else
		{
			scale = std::max(safe * pow(err, -alpha), min_scale);
			hh *= scale;
			reject = 1;
		}
	}

	// Calculates coefficients for dense output
	// neqn = number of equations
	// tt = initial time
	// tw = working time
	// yy = current state of the problem
	// yyp = current derivative of problem
	// yw = working state of the problem
	// yw2 = second working state of the problem
	// ywp = working derivative of the problem
	// k = Runge-kutta slope estimates
	// r = dense output coefficients
	void dense(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		double tt,
		double tw,
		double* yy,
		double* yyp,
		double* yw,
		double* yw2,
		double* ywp,
		double* k2,
		double* k3,
		double* k4,
		double* k5,
		double* k6,
		double* k7,
		double* k8,
		double* k9,
		double* k10,
		double* r1,
		double* r2,
		double* r3,
		double* r4,
		double* r5,
		double* r6,
		double* r7,
		double* r8,
		void* params)
	{
		double h = tw - tt;
		for (int i = 0; i < neqn; i++)
		{
			r1[i] = yy[i];
			r2[i] = yw[i] - yy[i];
			r3[i] = h * yyp[i] - r2[i];
			r4[i] = r2[i] - h * ywp[i] - r3[i];
			r5[i] = d41 * yyp[i] + d46 * k6[i] + d47 * k7[i] + d48 * k8[i] + d49 * k9[i] + d410 * k10[i] + d411 * k2[i] + d412 * k3[i];
			r6[i] = d51 * yyp[i] + d56 * k6[i] + d57 * k7[i] + d58 * k8[i] + d59 * k9[i] + d510 * k10[i] + d511 * k2[i] + d512 * k3[i];
			r7[i] = d61 * yyp[i] + d66 * k6[i] + d67 * k7[i] + d68 * k8[i] + d69 * k9[i] + d610 * k10[i] + d611 * k2[i] + d612 * k3[i];
			r8[i] = d71 * yyp[i] + d76 * k6[i] + d77 * k7[i] + d78 * k8[i] + d79 * k9[i] + d710 * k10[i] + d711 * k2[i] + d712 * k3[i];
		}
		for (int i = 0; i < neqn; i++)
			yw2[i] = yy[i] + h * (a141 * yyp[i] + a147 * k7[i] + a148 * k8[i] + a149 * k9[i] + a1410 * k10[i] + a1411 * k2[i] + a1412 * k3[i] + a1413 * ywp[i]);
		f(tt + c14 * h, yw2, k10, params);
		for (int i = 0; i < neqn; i++)
			yw2[i] = yy[i] + h * (a151 * yyp[i] + a156 * k6[i] + a157 * k7[i] + a158 * k8[i] + a1511 * k2[i] + a1512 * k3[i] + a1513 * ywp[i] + a1514 * k10[i]);
		f(tt + c15 * h, yw2, k2, params);
		for (int i = 0; i < neqn; i++)
			yw2[i] = yy[i] + h * (a161 * yyp[i] + a166 * k6[i] + a167 * k7[i] + a168 * k8[i] + a169 * k9[i] + a1613 * ywp[i] + a1614 * k10[i] + a1615 * k2[i]);
		f(tt + c16 * h, yw2, k3, params);
		for (int i = 0; i < neqn; i++)
		{
			r5[i] = h * (r5[i] + d413 * ywp[i] + d414 * k10[i] + d415 * k2[i] + d416 * k3[i]);
			r6[i] = h * (r6[i] + d513 * ywp[i] + d514 * k10[i] + d515 * k2[i] + d516 * k3[i]);
			r7[i] = h * (r7[i] + d613 * ywp[i] + d614 * k10[i] + d615 * k2[i] + d616 * k3[i]);
			r8[i] = h * (r8[i] + d713 * ywp[i] + d714 * k10[i] + d715 * k2[i] + d716 * k3[i]);
		}
	}

	// Interpolates dense output
	// neqn = number of equations
	// t = initial time
	// tout = final desired integration time
	// y = initial state of the problem
	// yp = derivative of problem
	// tt = initial time
	// tw = working time
	// r = dense output coefficients
	void intrp(
		int neqn,
		double& t,
		double tout,
		std::vector<double>& y,
		std::vector<double>& yp,
		double tt,
		double tw,
		double* r1,
		double* r2,
		double* r3,
		double* r4,
		double* r5,
		double* r6,
		double* r7,
		double* r8)
	{
		t = tout;
		double h = tt - tw;
		double s = (tout - tw) / h;
		double s1 = 1.0 - s;

		for (int i = 0; i < neqn; i++)
		{
			double a6 = r7[i] + s * r8[i];
			double a5 = r6[i] + a6 * s1;
			double a4 = r5[i] + a5 * s;
			double a3 = r4[i] + a4 * s1;
			double a2 = r3[i] + a3 * s;
			double a1 = r2[i] + a2 * s1;

			y[i] = r1[i] + s * a1;
			yp[i] = 1.0 / h * (a1 - s * (a2 - s1 * (a3 - s * (a4 - s1 * (a5 - s * (a6 - s1 * r8[i]))))));
		}
	}
}