
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#include <iostream>
#include <thread>
#include <chrono>
#include <cuda.h>
#include <stdlib.h>

#define NUM_TEAMS 400
#define RUNNER_COUNT 4
#define DISTANCE 400


/*
* 1 ile 5 arasında rastgele bir float sayı üretir.
* GPU'dan çalıştırmak için __device__ kullanır ve GPU'dan da çağrılacaktır.
*/
__device__ float generateRandomNumber(curandState* state) {

    return (curand_uniform(state) * 4 + 1);
}
/*
* Bayrak yarışı için algoritma.
* Ayrıca:
    - Her takım için kaç saniyenin geçtiğini hesaplar,
    - Herhangi bir koşucunun bitiş sırasını
    - Anlık olarak o anki takımın hangi koşuşunun koştuğunu  (1. 2. 3. veya 4.)
    - Ve koşucuların katettiği mesafeyi de hesaplar

  finishedOrder kaç takımın yarışı bitirdiğini hesaplamak içindir.
*
* GPU'dan çalıştırmak için __global__ kullanır ve CPU'dan çağrılır.
*/
__global__ void race(float* distances, int* currentRunner, int* placements, int* finishedOrder, int* seconds) {

    int index = threadIdx.x;

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(0, threadId, 0, &state);

    float speed = generateRandomNumber(&state);


    if (distances[index] < DISTANCE) {

        if (distances[index] + speed >= DISTANCE) {
            distances[index] = DISTANCE;
            placements[index] = finishedOrder[0] + 1;
            finishedOrder[0]++;
        }
        else {
            distances[index] += speed;
        }


        // Bayrak degisimi icin yazılmıs fonksiyon
        int checkpoint = (DISTANCE / RUNNER_COUNT) * currentRunner[index];

        if (distances[index] >= checkpoint) {

            if (currentRunner[index] < RUNNER_COUNT)
                currentRunner[index]++;
        }
        seconds[index]++;

    }

}


/*
* Takımların kat ettiği mesafelerden yarışın bitip bitmediğini kontrol eder.
*/
bool isRaceFinished(float* distances)
{
    for (int i = 0; i < NUM_TEAMS; i++) {
        if (distances[i] < DISTANCE) {
            return false;
        }
    }
    return true;
}


/*
* Şu anda yarışan her takımın mesafelerini yazdırır.
* printAll false ise sadece "takımCount" adet kosucuyu, "takımKosucuları" parametresinden çekerek yazdırır.
* aksi halde tüm kosucuları yazdırır.
*/
void printDistances(float* distances, int* currentRunner, int takimCount, int* takimSiralari, int* placements, int* seconds, bool printAll) {
    int takimIndex = 0;
    int minutes;
    int remainingSeconds;
    for (int i = 0; i < NUM_TEAMS; i++) {
        if (printAll)
        {
            if (placements[i] != 0) {
                if (seconds[i] > 60) {
                    minutes = seconds[i] / 60;
                    remainingSeconds = seconds[i] % 60;
                    std::cout << "Team " << i + 1 << ": Runner " << currentRunner[i] << " - " << " Finished at: " << placements[i] << "th place. " << " with " << minutes << " minutes " << remainingSeconds << ".seconds. \n" << std::endl;
                }
                else {
                    std::cout << "Team " << i + 1 << ": Runner " << currentRunner[i] << " - " << " Finished at: " << placements[i] << "th place. " << " with " << seconds[i] << " seconds.\n" << std::endl;
                }
            }
        }
        else
        {
            if (takimIndex < takimCount) {
                if (i + 1 == takimSiralari[takimIndex]) {
                    if (placements[i] != 0) {
                        if (seconds[i] > 60) {
                            minutes = seconds[i] / 60;
                            remainingSeconds = seconds[i] % 60;
                            std::cout << "Team " << i + 1 << ": Runner " << currentRunner[i] << " - " << " Finished at: " << placements[i] << "th place. " << " with " << minutes << " minutes " << remainingSeconds << ".seconds. \n";
                        }
                        else {
                            std::cout << "Team " << i + 1 << ": Runner " << currentRunner[i] << " - " << " Finished at: " << placements[i] << "th place. " << " with " << seconds[i] << " seconds. \n " << std::endl;
                        }
                    }
                    else {
                        std::cout << "Team " << i + 1 << ": Runner " << currentRunner[i] << " - " << distances[i] << " m's of distance traveled with speed of: " << (distances[i] / seconds[i]) << "m/s" << " with " << seconds[i] << ".seconds. \n" << std::endl;
                    }
                    takimIndex++;
                }
            }
        }
    }
}

/*
* Yarışı bitiren her takımın sıralamasını yazdırır.
* Parametre olarak geçirilen "takımCount" kadar koşucuyu "takımSıraları"'ndan çekerek yazdırır.
* Ayrıca yarışı bitirmeleri için kaç saniye yarıştıklarını de yazdırır.
*/
void printTeamsPlacements(int takimCount, int* takimSiralari, int* placements, int* seconds) {

    int takimIndex = 0;
    for (int i = 0; i < NUM_TEAMS; i++) {

        if (takimIndex < takimCount) {

            if (i + 1 == takimSiralari[takimIndex]) {

                if (seconds[i] != 0) {
                    if (seconds[i] > 60) {
                        int minutes = seconds[i] / 60;
                        int remainingSeconds = seconds[i] % 60;
                        std::cout << "Team " << i + 1 << " -- Finished at: " << placements[i] << "th place. With " << minutes << " minutes " << remainingSeconds << ".seconds.\n";
                    }
                    else {
                        std::cout << "Team " << i + 1 << " -- Finished at: " << placements[i] << "th place. With " << seconds[i] << " seconds.\n" << std::endl;

                    }
                }
                takimIndex++;
            }
        }
    }
}

int main() {


    float* distances;
    int* currentRunner;
    int* placements;
    int* seconds;
    int* finishedOrder;

    cudaMallocManaged(&finishedOrder, sizeof(int));
    cudaMallocManaged(&distances, NUM_TEAMS * sizeof(float));
    cudaMallocManaged(&currentRunner, NUM_TEAMS * sizeof(int));
    cudaMallocManaged(&seconds, NUM_TEAMS * sizeof(int));
    cudaMallocManaged(&placements, NUM_TEAMS * sizeof(int));


    finishedOrder[0] = 0;


    std::cout << "Kac takimi takip etmek istiyorsunuz: ";
    int takimCount;

    std::cin >> takimCount;

    int* takimlar;
    cudaMallocManaged(&takimlar, NUM_TEAMS * sizeof(int));

    std::cout << "Takip etmek istediğiniz takımların no'sunu giriniz: ";
    for (int i = 0; i < takimCount; i++) {
        std::cin >> takimlar[i];
        while (takimlar[i] < 0 || takimlar[i] > 400) {
            std::cout << "1 ila 400 arası (1 ile 400 dahil) bir takım numarası giriniz: ";
            std::cin >> takimlar[i];
        }
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    for (int i = 0; i < NUM_TEAMS; i++) {
        distances[i] = 0.f;
        currentRunner[i] = 1;
        placements[i] = 0;
        seconds[i] = 0;
    }

    while (!isRaceFinished(distances)) {
        race << <1, NUM_TEAMS >> > (distances, currentRunner, placements, finishedOrder, seconds);
        cudaDeviceSynchronize();
        printDistances(distances, currentRunner, takimCount, takimlar, placements, seconds, false);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        //system("CLS");
    }


    printDistances(distances, currentRunner, takimCount, takimlar, placements, seconds, true);

    std::cout << "\n\nYour teams placements: " << std::endl << std::endl;

    printTeamsPlacements(takimCount, takimlar, placements, seconds);


    cudaFree(distances);
    cudaFree(currentRunner);

    return 0;
}