#define STRINGIFY(A) #A

const char *vertexShader = STRINGIFY(uniform float pointRadius; uniform float pointScale; uniform float densityScale; 
uniform float densityOffset; uniform vec3 lightPositionIn; varying vec3 posEye; varying vec3 normal; varying vec3 worldPos; varying vec3 lightlocation;

void main()
{
	posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
	float dist = length(posEye);
	//lightlocation = vec3(gl_ModelViewProjectionMatrix * vec4(lightPositionIn.xyz, 1.0));
	worldPos = gl_Vertex.xyz;

	//normal = gl_Normal.xyz;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
	normal = normalize(gl_NormalMatrix*gl_Normal);
	normal = (normal + 1.0f) / 2.0f;
	gl_FrontColor = vec4(1.0f,1.0f,1.0f, 1.0f);

}
);

const char *spherePixelShader = STRINGIFY(varying vec3 posEye; varying vec3 normal; uniform float near; uniform float far; uniform float color; 
uniform float pointRadius; varying vec3 worldPos; varying vec3 lightlocation;
	void main()
{
	//const vec3 lightDir = vec3(0.577, 0.577, 0.577);
	////const vec3 lightDir = vec3(0.0, 1.0, 0.0);

	//vec3 N;
	//N.xy = gl_TexCoord[0].st*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	//float mag = dot(N.xy, N.xy);
	//if (gl_Color.w == 100.0f){
	//	gl_FragColor = gl_Color;
	//}
	//else{
	//	//if (mag > 1.0) discard;
	//	//if (gl_Color.w == 0.0f)discard;

	//	N.z = sqrt(1.0 - mag);

	//	float diffuse = max(0.0, dot(lightDir, N));
	//	//gl_Color.w = 1.0f;

	//	//gl_FragColor = gl_Color;
	//	gl_FragColor.xyz = -normal;
	//	gl_FragColor.w = 1.0f;
	//	//gl_FragColor = vec4(N, gl_Color.w);

	//	//gl_FragColor = gl_Color * diffuse;
	//}
	
	//const vec3 lightDir = vec3(0.577, 0.577, 0.577);
	//const vec3 lightDir = vec3(0.0, 1.0, 0.0);
	//float diffuse = max(0.0, dot(lightDir, normal));
	//gl_FragColor.xyz = gl_Color.xyz * diffuse;
	//gl_FragColor.xyz = gl_Color.xyz;
	//gl_FragColor.xyz += vec3(0.1,0.1,0.1);

	//gl_FragColor.xyz = -gl_Normal.xyz;


	vec4 diffuse = gl_Color;

	const vec3 lightPosition = vec3(80.0f, 32.0f, 80.0f);
	//const vec3 lightPosition = lightlocation;
	const vec3 ambient = vec3(0.1f, 0.1f, 0.1f);
	const float lightRadius = 120.0f;
	const vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);
	vec3 incident = normalize(lightPosition - worldPos);
	float lambert = max(0.0, dot(incident, normal));
	float dist = length(lightPosition - worldPos);
	float atten = 1.0 - clamp(dist / lightRadius, 0.0, 1.0);
	vec3 viewDir = normalize(posEye - worldPos);
	vec3 halfDir = normalize(incident + viewDir);
	float rFactor = max(0.0, dot(halfDir, normal));
	float sFactor = pow(rFactor, 50.0);


	vec3 color = (diffuse.xyz*lightColor.xyz);
	color += (lightColor.xyz*sFactor)*0.33;
	gl_FragColor = vec4(color*atten*lambert, diffuse.w);
	gl_FragColor.w = 1.0f;
	//gl_FragColor += (diffuse.xyz*lightColor.xyz)*0.1f;











	//gl_FragColor = gl_Color;
	//gl_FragColor.w = 0.2f;
}
);


const char *floorVS = STRINGIFY(
	varying vec4 vertexPosEye;  // vertex position in eye space  \n
varying vec3 normalEye;                                      \n
void main()                                                  \n
{
	\n
	gl_Position = gl_ModelViewProjectionMatrix *gl_Vertex;  \n
	gl_TexCoord[0] = gl_MultiTexCoord0;                      \n
	//vertexPosEye = gl_ModelViewMatrix *gl_Vertex;           \n
	//normalEye = gl_NormalMatrix *gl_Normal;                 \n
	gl_FrontColor = gl_Color;                                \n
}                                                            \n
);

const char *floorPS = STRINGIFY(
	uniform vec3 lightPosEye; // light position in eye space                      \n
uniform vec3 lightColor;                                                      \n
uniform sampler2D tex;                                                        \n
uniform sampler2D shadowTex;                                                  \n
varying vec4 vertexPosEye;  // vertex position in eye space                   \n
varying vec3 normalEye;                                                       \n
void main()                                                                   \n
{
	\n
	vec4 shadowPos = gl_TextureMatrix[0] * vertexPosEye;                      \n
	vec4 colorMap = texture2D(tex, gl_TexCoord[0].xy);                       \n

	//vec3 N = normalize(normalEye);                                            \n
	//vec3 L = normalize(lightPosEye - vertexPosEye.xyz);                       \n
	//float diffuse = max(0.0, dot(N, L));                                      \n

	//vec3 shadow = vec3(1.0) - texture2DProj(shadowTex, shadowPos.xyw).xyz;    \n

	//if (shadowPos.w < 0.0) shadow = lightColor;                               \n // avoid back projections
	//	gl_FragColor = vec4(gl_Color.xyz *colorMap.xyz *diffuse * lightColor, 1.0); \n
	gl_FragColor = colorMap; \n
}                                                                             \n
);

const char *mblurVS = STRINGIFY(
	uniform float timestep;                                    \n
	void main()                                                \n
{
	\n
	vec3 pos = gl_Vertex.xyz;                           \n
	vec3 vel = gl_MultiTexCoord0.xyz;                   \n
	vec3 pos2 = (pos - vel*timestep).xyz;                \n // previous position \n
	
	gl_Position = gl_ModelViewMatrix * vec4(pos, 1.0);  \n // eye space
	gl_TexCoord[0] = gl_ModelViewMatrix * vec4(pos2, 1.0); \n

	// aging                                                 \n
	float lifetime = gl_MultiTexCoord0.w;                    \n
	float age = gl_Vertex.w;                                 \n
	float phase = (lifetime > 0.0) ? (age / lifetime) : 1.0; \n // [0, 1]

	gl_TexCoord[1].x = phase;                                \n
	float fade = 1.0 - phase;                                \n
	//  float fade = 1.0;                                        \n

	//    gl_FrontColor = gl_Color;                              \n
	//gl_FrontColor = vec4(gl_Color.xyz, gl_Color.w*fade);  

	//gl_FrontColor = vec4(gl_Color.xyz, 0.9f);     \n

	float temp = gl_MultiTexCoord0.w ;
	if (temp < 2.5f){
		gl_FrontColor = vec4(gl_Color.xyz, 0.1f);     \n
	}
	else{
		gl_FrontColor = vec4(gl_Color.xyz, 0.9f);
	}
	/*if (temp<=0.2f ){
		gl_Color.x = 1.0f;
		gl_FrontColor = vec4(gl_Color.xyz, temp);     \n
	}*/

	/*if (temp <= 0.142f){
		gl_FrontColor = vec4(gl_Color.xyz, 0.0f);     \n
	}*/
}                                                            \n
);

const char *mblurGS =
"#version 120\n"
"#extension GL_EXT_geometry_shader4 : enable\n"
STRINGIFY(
uniform float pointRadius;  // point size in world space       \n
void main()                                                    \n
{
	\n
	// aging                                                   \n
	float phase = gl_TexCoordIn[0][1].x;                       \n
	float radius = pointRadius;                                \n
	//float temp = gl_FrontColorIn[0].w;
	//temp = temp / 0.9f;
	//radius = temp;
	// eye space                                               \n
	vec3 pos = gl_PositionIn[0].xyz;                           \n
		vec3 pos2 = gl_TexCoordIn[0][0].xyz;                       \n
	vec3 motion = pos - pos2;                                  \n
	vec3 dir = normalize(motion);                              \n
	float len = length(motion);                                \n

	vec3 x = dir *radius;                                     \n
	vec3 view = normalize(-pos);                               \n
	vec3 y = normalize(cross(dir, view)) * radius;             \n
	float facing = dot(view, dir);                             \n

	// check for very small motion to avoid jitter             \n
	float threshold = 0.01;                                    \n

	

	if ((len < threshold) || (facing > 0.95) || (facing < -0.95))
	{
		\n
			pos2 = pos;
		\n
			x = vec3(radius, 0.0, 0.0);
		\n
			y = vec3(0.0, -radius, 0.0);
		\n
			//x = x*0.5*temp+x*0.5 ;
			//y = y*0.5*temp+y*0.5 ;
			//x = x*temp;
		//y = y*temp;
	}                                                          \n

		// output quad                                             \n
		gl_FrontColor = gl_FrontColorIn[0];                        \n
	gl_TexCoord[0] = vec4(0, 0, 0, phase);                     \n
	gl_TexCoord[1] = gl_PositionIn[0];                         \n
	gl_Position = gl_ProjectionMatrix * vec4(pos + x + y, 1);  \n
	EmitVertex();                                              \n

	gl_TexCoord[0] = vec4(0, 1, 0, phase);                     \n
	gl_TexCoord[1] = gl_PositionIn[0];                         \n
	gl_Position = gl_ProjectionMatrix * vec4(pos + x - y, 1);  \n
	EmitVertex();                                              \n

	gl_TexCoord[0] = vec4(1, 0, 0, phase);                     \n
	gl_TexCoord[1] = gl_PositionIn[0];                         \n
	gl_Position = gl_ProjectionMatrix * vec4(pos2 - x + y, 1); \n
	EmitVertex();                                              \n

	gl_TexCoord[0] = vec4(1, 1, 0, phase);                     \n
	gl_TexCoord[1] = gl_PositionIn[0];                         \n
	gl_Position = gl_ProjectionMatrix * vec4(pos2 - x - y, 1); \n
	EmitVertex();                                              \n
}                                                              \n
);

const char *particlePS = STRINGIFY(
	uniform float pointRadius;                                         \n
	void main()                                                        \n
{
	\n
	// calculate eye-space sphere normal from texture coordinates  \n
	vec3 N;                                                        \n
	N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);    \n
	float r2 = dot(N.xy, N.xy);                                    \n

	if (r2 > 1.0) discard;   // kill pixels outside circle         \n
	N.z = sqrt(1.0 - r2);                                            \n

		//  float alpha = saturate(1.0 - r2);                              \n
		float alpha = clamp((1.0 - r2), 0.0, 1.0);                     \n
		alpha *= gl_Color.w;                                           \n

		//gl_FragColor = vec4(gl_Color.xyz * alpha, alpha);              \n
		gl_FragColor = vec4(N , 1.0);
}                                                                  \n
);


const char *passThruVS = STRINGIFY(
	varying vec3 posEye;
	void main()                                                        \n
{
	\n
	gl_Position = gl_Vertex;                                       \n
	gl_TexCoord[0] = gl_MultiTexCoord0;                            \n
	posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));

	//gl_FrontColor = gl_Color;                                      \n
}                                                                  \n
);

const char *texture2DPS = STRINGIFY(
	uniform sampler2D tex;                                             \n
	void main()                                                        \n
{
	\n
	gl_FragColor = texture2D(tex, gl_TexCoord[0].xy);              \n
}                                                                  \n
);

const char *postprocessingPS = STRINGIFY(
	uniform sampler2D tex;                                             \n
	uniform float isvertical;
    uniform vec2 pixelSize;
	varying vec3 posEye;
void main()                                                        \n
{
	\n
	float weight[5];
	/*weight[0] = 0.12f;
	weight[1] = 0.22f;
	weight[2] = 0.32f;
	weight[3] = 0.22f;
	weight[4] = 0.12f;	*/

	weight[0] = 0.208f;
	weight[1] = 0.208f;
	weight[2] = 0.208f;
	weight[3] = 0.208f;
	weight[4] = 0.208f;

	vec2 values[5];

	float dist = length(posEye);
	float rate = 1.0f / ((dist + 10.0f) / 240.0f);

	if (isvertical == 1.0f) {
		values[0] = vec2(0.0, -pixelSize.y * 1.0005*rate);
		values[1] = vec2(0.0, -pixelSize.y * 1.00025*rate);
		values[2] = vec2(0.0, pixelSize.y);
		values[3] = vec2(0.0, pixelSize.y * 1.00025*rate);
		values[4] = vec2(0.0, pixelSize.y * 1.0005*rate);
	}
	else {
		values[0] = vec2(-pixelSize.x * 1.0005*rate, 0.0);
		values[1] = vec2(-pixelSize.x * 1.00025*rate, 0.0);
		values[2] = vec2(pixelSize.x, 0.0);
		values[3] = vec2(pixelSize.x * 1.00025*rate, 0.0);
		values[4] = vec2(pixelSize.x * 1.0005*rate, 0.0);
	}

	if (texture2D(tex, gl_TexCoord[0].xy).x > 0.1f){
		for (int i = 1; i < 4; i++){
			vec4 tmp = texture2D(tex, gl_TexCoord[0].xy + values[i]);
			gl_FragColor += tmp * (weight[i]);
		}
	}
	else{
		for (int i = 0; i < 5; i++) {
			vec4 tmp = texture2D(tex, gl_TexCoord[0].xy + values[i]);
			gl_FragColor.xyz += tmp.xyz * (weight[i]);
		}
	}

}                                                                  \n
);