#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#define threshold_minor 90
#define eps 1e-8

using namespace std;

int main(int argc,char *argv[])
{
		if (argc < 3) {
				printf("\n Need an input file and an output file\n");
				return 0;
		}
		int i, j, k, n, m;

		double MRMC = 0, CBP = 0, TBWDC = 0, rate_i = 0, normal_BW = 0, intensive_BW = 0, PBW = 0;
		int flag_minor=0, flag_normal=0, flag_intensive=0;
		FILE * fp = fopen(argv[1],"r");
		FILE * output = fopen(argv[2],"w");
		vector <double> standaloneBW, externalBW;
		vector <vector<double> > achievedBW;
		vector <vector<double> > achieved_relative_speed;


		fscanf(fp, "%d", &n);
		achievedBW.resize(n);
		achieved_relative_speed.resize(n);
		for (i = 0; i < n; ++i)
		{
				double tmp;
				fscanf(fp, "%lf", &tmp);
				standaloneBW.push_back(tmp);
		}

		fscanf(fp, "%d", &m);
		for (i = 0; i < m; ++i)
		{
				double tmp;
				fscanf(fp, "%lf", &tmp);
				externalBW.push_back(tmp);
		}

		for (i = 0; i < n; ++i)
				for (j = 0; j < m; ++j)
				{
						double tmp;
						fscanf(fp, "%lf", &tmp);
						achievedBW[i].push_back(tmp);
						PBW = max(PBW, tmp);
				}

		// calculate achieved_relative_speed percentage

		for (i = 0; i < n; ++i)
				for (j = 0; j < m; ++j)
				{
						achieved_relative_speed[i].push_back(achievedBW[i][j]/standaloneBW[i]*100);
				}

		// determine the boundary between minor region and normal region
		// first branch is for no minor region case
		// second branch is to find the
		if (achieved_relative_speed[0][m-1] < threshold_minor) {flag_minor = -1; MRMC = -1; normal_BW = 0; printf("here\n");}
		else
		{
				double reduction = min(100 - achieved_relative_speed[0][m-1], 95.0);
				int l, normal_boundary, intensive_boundary;
				for (i = 0; i < n; ++i)
				{
						if (reduction * 2 < (100-achieved_relative_speed[i][m-1])) break;
				}
				normal_boundary = i; normal_BW = standaloneBW[i]; MRMC = 100- achieved_relative_speed[i-1][m-1];
				for (j = 0; j < m; ++j)
				{
						if ((100-achieved_relative_speed[i][j]) >= reduction * 2) break;
				}
				TBWDC = standaloneBW[i]+externalBW[j];

				for (k = i; k < n; ++k)
				{
						if ((100-achieved_relative_speed[k][0] >= reduction * 2)) break;
				}
				if (k == n) {flag_intensive = -1; intensive_BW=PBW; intensive_boundary=n;}
				else { intensive_BW=standaloneBW[k]; intensive_boundary = k;}
				vector <int> balancepoints(m,0);
				for (i = normal_boundary; i < intensive_boundary; ++i)
				{
						double sum = 0.0, cur; int cnt = 0;
						for (j = 1; j < m; ++j)
						{
								if (standaloneBW[i] + externalBW[j] >= TBWDC)
								{
										cur = (achieved_relative_speed[i][j-1]-achieved_relative_speed[i][j])/(externalBW[j]-externalBW[j-1]);
										if (cnt != 0) {
												if (cur * 3 < sum/cnt) break;
										}
										sum+= cur; cnt++;
								}
						}
						balancepoints[j]++;
				}
				double sum = 0.0; int cnt = 0;
				for (j = 1; j < m; ++j)
						sum+=balancepoints[j]*externalBW[j];
				CBP=sum/(intensive_boundary - normal_boundary+1);
				double rate_sum=0.0; int rate_cnt = 0;
				for (i = normal_boundary; i < intensive_boundary; ++i)
						for (j = 1; j < m; ++j)
								if (standaloneBW[i]+externalBW[j]>=TBWDC && externalBW[j]<=CBP)
								{
										rate_sum+=(achieved_relative_speed[i][j-1]-achieved_relative_speed[i][j])/(externalBW[j]-externalBW[j-1]);
										rate_cnt++;
								}

				rate_i = rate_sum/rate_cnt;

		}

		fprintf(output, "Normal BW %lf\n", normal_BW);
		fprintf(output, "intensive BW %lf\n", intensive_BW);
		fprintf(output, "MRMC %lf\n", MRMC);
		fprintf(output, "TBWDC %lf\n", TBWDC);
		fprintf(output, "PBW %lf\n", PBW);
		fprintf(output, "CBP %lf\n", CBP);
		fprintf(output, "rate_i %lf\n", rate_i);

		return 0;

}
