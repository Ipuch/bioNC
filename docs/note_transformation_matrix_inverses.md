# Transformation matrices $B$ and their analytical inverses

###  Natural segment parameters

$$\begin{aligned}
L_i&=\sqrt{\left(\mathbf{r}_{P_i}-\mathbf{r}_{D_i}\right)^2}
&\beta_i&=\cos^{-1}\left(\mathbf{u}_i\bullet\mathbf{w}_i\right)\\[2pt]
\alpha_i&=\cos^{-1}\left(\frac{\mathbf{r}_{P_i}-\mathbf{r}_{D_i}}{L_i}\bullet\mathbf{w}_i\right)
&\gamma_i&=\cos^{-1}\left(\mathbf{u}_i\bullet\frac{\mathbf{r}_{P_i}-\mathbf{r}_{D_i}}{L_i}\right)
\end{aligned}$$

The columns of $\mathbf{B}_i$ are $\left(\mathbf{u}_i,\ \mathbf{r}_{P_i}-\mathbf{r}_{D_i},\ \mathbf{w}_i\right)$ expressed in the orthonormal SCS. The index $i$ is dropped below.

### The $\mathbf{B}$ matrices I have implemented in BioNC


$$\mathbf{B}_{uv}=\begin{pmatrix}1&L\cos\gamma&\cos\beta\\0&L\sin\gamma&(\cos\alpha-\cos\beta\cos\gamma)/\sin\gamma\\0&0&\delta/\sin\gamma\end{pmatrix}$$

$$\mathbf{B}_{vu}=\begin{pmatrix}\sin\gamma&0&(\cos\beta-\cos\alpha\cos\gamma)/\sin\gamma\\\cos\gamma&L&\cos\alpha\\0&0&\delta/\sin\gamma\end{pmatrix}$$

$$\mathbf{B}_{wu}=\begin{pmatrix}\sin\beta&L(\cos\gamma-\cos\alpha\cos\beta)/\sin\beta&0\\0&L\delta/\sin\beta&0\\\cos\beta&L\cos\alpha&1\end{pmatrix}$$

$$\mathbf{B}_{uw}=\begin{pmatrix}1&L\cos\gamma&\cos\beta\\0&L\delta/\sin\beta&0\\0&L(\cos\alpha-\cos\beta\cos\gamma)/\sin\beta&\sin\beta\end{pmatrix}$$

where,
$$\delta=\sqrt{1-\cos^{2}\alpha-\cos^{2}\beta-\cos^{2}\gamma+2\cos\alpha\cos\beta\cos\gamma}$$

### Link with the classical Dumas matrices

In the classical Lyon/Dumas implementations the square-root entries all have the form

$$E=1-\cos^{2}X-\left(\frac{\cos Y-\cos X\cos Z}{\sin Z}\right)^{2},\qquad \{X,Y,Z\}=\{\alpha,\beta,\gamma\}$$

Develop it. First put everything over $\sin^{2}Z$:

$$E=\frac{\left(1-\cos^{2}X\right)\sin^{2}Z-\left(\cos Y-\cos X\cos Z\right)^{2}}{\sin^{2}Z}$$

Then work on the numerator alone:

$$\begin{aligned}
N&=\left(1-\cos^{2}X\right)\sin^{2}Z-\left(\cos Y-\cos X\cos Z\right)^{2}\\[2pt]
&=\left(1-\cos^{2}X\right)\left(1-\cos^{2}Z\right)-\left(\cos Y-\cos X\cos Z\right)^{2}\\[2pt]
&=1-\cos^{2}X-\cos^{2}Z+\cos^{2}X\cos^{2}Z-\left(\cos Y-\cos X\cos Z\right)^{2}\\[2pt]
&=1-\cos^{2}X-\cos^{2}Z+\cos^{2}X\cos^{2}Z-\cos^{2}Y+2\cos X\cos Y\cos Z-\cos^{2}X\cos^{2}Z\\[2pt]
&=1-\cos^{2}X-\cos^{2}Y-\cos^{2}Z+2\cos X\cos Y\cos Z
\end{aligned}$$

The $\cos^{2}X\cos^{2}Z$ terms have cancelled, and the result is symmetric in the three angles. It no longer matters which of $\alpha,\beta,\gamma$ played $X$, $Y$ or $Z$. That last line is $\delta^{2}$, so

$$E=\frac{\delta^{2}}{\sin^{2}Z}\qquad\text{and}\qquad\sqrt{E}=\frac{\delta}{\sin Z}$$

Applied to $\mathbf{B}_{uv}$ and $\mathbf{B}_{wu}$:

$$\begin{aligned}
[2,2]:\quad&&\sqrt{1-\left(\cos\beta\right)^{2}-\left(\frac{\cos\alpha-\cos\beta\cos\gamma}{\sin\gamma}\right)^{2}}&=\frac{\delta}{\sin\gamma}\\[4pt]
[1,1]:\quad&&L\sqrt{1-\left(\cos\alpha\right)^{2}-\left(\frac{\cos\gamma-\cos\alpha\cos\beta}{\sin\beta}\right)^{2}}&=\frac{L\,\delta}{\sin\beta}
\end{aligned}$$

with $(X,Y,Z)=(\beta,\alpha,\gamma)$ and $(\alpha,\gamma,\beta)$, respectively. The symmetry hidden inside your square roots is exactly $\det\mathbf{G}$ below, which is why one $\delta$ serves all four types and all four inverses. $\mathbf{B}_{vu}$ and $\mathbf{B}_{uw}$ are the two remaining variants, obtained the same way.

### Check

Because the columns *are* $\mathbf{u}$, $\mathbf{r}_P-\mathbf{r}_D$, $\mathbf{w}$, every type factorises the same way:

$$\mathbf{B}^\top \mathbf{B}=\mathbf{D}\,\mathbf{G}\,\mathbf{D},\qquad \mathbf{D}=\operatorname{diag}(1,L,1),\qquad \mathbf{G}=\begin{pmatrix}1&\cos\gamma&\cos\beta\\\cos\gamma&1&\cos\alpha\\\cos\beta&\cos\alpha&1\end{pmatrix}$$

$\mathbf{G}$ is the Gram matrix of the unit triad $\left(\mathbf{u},\ (\mathbf{r}_P-\mathbf{r}_D)/L,\ \mathbf{w}\right)$ — it carries the angles alone. **The $\delta$ written above is exactly its determinant** — that is where it comes from:

$$\delta^2=\det\mathbf{G}=1-\cos^2\alpha-\cos^2\beta-\cos^2\gamma+2\cos\alpha\cos\beta\cos\gamma,\qquad \det\mathbf{B}=L\,\delta$$

For arbitrary $(\alpha,\beta,\gamma)$, $\mathbf{G}$ is symmetric with unit diagonal but need not be the Gram matrix of anything. Its leading $2\times2$ minor $1-\cos^2\gamma$ is never negative, so by Sylvester the sign of $\det\mathbf{G}$ decides on its own:

- $\delta^2>0$ — the triad is linearly independent, a genuine basis, $\mathbf{B}$ invertible;
- $\delta^2=0$ — the triad is coplanar, $\mathbf{B}$ singular;
- $\delta^2<0$ — **no** triad in $\mathbb{R}^3$ has these three pairwise angles; $(\alpha,\beta,\gamma)$ is not realisable.

So $\delta^2>0$ is *the* admissibility test on a segment's angles, type-independent, and worth running before anything else. It also subsumes the poles of the formulas above: $\delta^2>0$ forces $\alpha,\beta,\gamma\notin\{0,\pi\}$.

### Analytical inverses

We can also get the inverse to avoid numerical inversion, which is very practical for symbolic computing.

$$\mathbf{B}_{uv}^{-1}=\begin{pmatrix}1&-\cos\gamma/\sin\gamma&(\cos\alpha\cos\gamma-\cos\beta)/(\delta\sin\gamma)\\0&1/(L\sin\gamma)&(\cos\beta\cos\gamma-\cos\alpha)/(L\delta\sin\gamma)\\0&0&\sin\gamma/\delta\end{pmatrix}$$

$$\mathbf{B}_{vu}^{-1}=\begin{pmatrix}1/\sin\gamma&0&(\cos\alpha\cos\gamma-\cos\beta)/(\delta\sin\gamma)\\-\cos\gamma/(L\sin\gamma)&1/L&(\cos\beta\cos\gamma-\cos\alpha)/(L\delta\sin\gamma)\\0&0&\sin\gamma/\delta\end{pmatrix}$$

$$\mathbf{B}_{wu}^{-1}=\begin{pmatrix}1/\sin\beta&(\cos\alpha\cos\beta-\cos\gamma)/(\delta\sin\beta)&0\\0&\sin\beta/(L\delta)&0\\-\cos\beta/\sin\beta&(\cos\beta\cos\gamma-\cos\alpha)/(\delta\sin\beta)&1\end{pmatrix}$$

$$\mathbf{B}_{uw}^{-1}=\begin{pmatrix}1&(\cos\alpha\cos\beta-\cos\gamma)/(\delta\sin\beta)&-\cos\beta/\sin\beta\\0&\sin\beta/(L\delta)&0\\0&(\cos\beta\cos\gamma-\cos\alpha)/(\delta\sin\beta)&1/\sin\beta\end{pmatrix}$$

### Generalized inverse

$\mathbf{B}$ is square and invertible whenever $\delta^2>0$, so its Moore–Penrose pseudo-inverse coincides with $\mathbf{B}^{-1}$, and the normal-equation form collapses to a single expression valid for *every* type:

$$\mathbf{B}^{-1}=\mathbf{B}^{+}=\left(\mathbf{B}^\top\mathbf{B}\right)^{-1}\mathbf{B}^\top=\mathbf{D}^{-1}\mathbf{G}^{-1}\mathbf{D}^{-1}\mathbf{B}^\top$$

$$\mathbf{G}^{-1}=\frac{1}{\delta^2}\begin{pmatrix}1-\cos^2\alpha&\cos\alpha\cos\beta-\cos\gamma&\cos\gamma\cos\alpha-\cos\beta\\\cos\alpha\cos\beta-\cos\gamma&1-\cos^2\beta&\cos\beta\cos\gamma-\cos\alpha\\\cos\gamma\cos\alpha-\cos\beta&\cos\beta\cos\gamma-\cos\alpha&1-\cos^2\gamma\end{pmatrix}$$

$\mathbf{D}$ and $\mathbf{G}$ depend only on $(L,\alpha,\beta,\gamma)$, never on the type, so this returns the inverse of any $\mathbf{B}$, with no new derivation. 
The four closed forms above are what it reduces to.

Derived with sympy; matches numerical inversion to $3\cdot10^{-11}$ over $\sim10^4$ random admissible $(L,\alpha,\beta,\gamma)$.
